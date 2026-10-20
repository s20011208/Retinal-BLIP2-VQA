"""Faithful public port of the image-balanced P1 optimizer and resumable sampler.

One uniformly chosen QA per original-image group per epoch; group-balanced
microbatch-one accumulation; AdamW; linear decay; validation-only early stopping.
Frozen OPT stays graph-enabled so its input Jacobian reaches the visual adapter.
The public port has a new code fingerprint: historical checkpoints support
inference, not silent optimizer continuation under different code/data formats.
"""
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .data import grouped_budget, load_qa
from .evaluate import evaluate_model, image_for
from .io import atomic_json, digest, load_config, now, object_hash, run_lock
from .models.blip2_adapter import MODEL_REVISION, RetinalBLIP2

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def training_code_hash():
    root = Path(__file__).parent
    return object_hash({str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*.py'))})



def bind_training_sources(rows, model_path):
    """Hash actual admitted image and local base-model bytes, not path names/mtime."""
    image_hashes = {}
    for row in rows:
        path = str(Path(row['image_path']).resolve())
        if path not in image_hashes:
            image_hashes[path] = digest(path)
        if row.get('raw_sha256') and row['raw_sha256'] != image_hashes[path]:
            raise ValueError('actual image bytes differ from the supplied identity hash')
    return {'image_files_sha256':image_hashes,'base_model_files_sha256':bind_model_sources(model_path),
            'hash_scope':'selected training + fixed validation image bytes; local model weights/config/tokenizer files'}


def bind_model_sources(model_path):
    root = Path(model_path).resolve()
    if not root.is_dir() or not (root/'config.json').is_file():
        raise ValueError('a local base-model directory containing config.json is required')
    files = [path for path in root.rglob('*') if path.is_file() and
             not any(part.startswith('.') for part in path.relative_to(root).parts) and
             path.suffix in ('.json','.safetensors','.bin','.txt','.model','.tiktoken')]
    if not any(path.suffix in ('.safetensors','.bin') for path in files):
        raise ValueError('local base-model directory has no weight payload')
    return {str(path.relative_to(root)):digest(path) for path in sorted(files)}


def validate_config(config):
    for name, default, zero in (('learning_rate', None, False), ('weight_decay', .01, True),
                                ('clip_norm', 1., False), ('timeout_seconds', 10800, False)):
        value = config.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
                value < 0 or (value == 0 and not zero)):
            raise ValueError('invalid numeric training setting: ' + name)
    for name, default in (('effective_batch', 16), ('max_epochs', 20), ('patience', 4), ('checkpoint_interval', 10)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('positive integer required: ' + name)
    seed = config.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0,2**32)')
    budget = config.get('budget')
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= 1:
        raise ValueError('budget must be in (0,1]')
    if config.get('mode') not in ('B1', 'M1'):
        raise ValueError('training requires B1 or M1')
    for field in ('identities', 'answers', 'model_path', 'output_dir'):
        if not isinstance(config.get(field), str) or not config[field]:
            raise ValueError('missing configured path: ' + field)


def _train(config, resume=False):
    global STOP
    STOP = False
    validate_config(config)
    seed, device = config['seed'], config.get('device', 'cuda')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows, admission = load_qa(config['identities'], config['answers'], ('train', 'val'), config.get('data_root'))
    train_rows, budget = grouped_budget([row for row in rows if row['split'] == 'train'], config['budget'], seed)
    val_rows = [row for row in rows if row['split'] == 'val']
    if not val_rows:
        raise ValueError('fixed validation role is required')
    groups = {}
    for row in train_rows:
        groups.setdefault(row['group_id'], []).append(row)
    out = Path(config['output_dir'])
    checkpoint_dir, result_dir = out / 'checkpoints', out / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    latest, best_path = checkpoint_dir / 'latest.pt', checkpoint_dir / 'best.pt'
    if best_path.exists() and not latest.exists():
        raise FileExistsError('selected adapter exists without resumable state; preserve it and use a new run directory')
    if latest.exists() and not resume:
        raise FileExistsError('run exists; use --resume instead of overwriting it')
    if resume and not latest.exists():
        raise FileNotFoundError('--resume requested but latest checkpoint is absent')
    data_hash = object_hash({key: admission[key] for key in ('identities_sha256', 'answers_sha256')})
    source_binding = bind_training_sources(train_rows + val_rows, config['model_path'])
    source_hash = object_hash(source_binding)
    config_hash, code_hash = object_hash(config), training_code_hash()
    saved = None
    if resume:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved.get('data_hash') != data_hash or saved.get('config_hash') != config_hash:
            raise ValueError('resume requires identical admitted data and resolved config')
        if saved.get('source_content_hash') != source_hash:
            raise ValueError('training image or base-model bytes changed; cannot resume')
        if saved.get('training_code_hash') != code_hash:
            raise ValueError('training implementation changed; validate a migration or start a separate run')
    model = RetinalBLIP2.from_pretrained(config['model_path'], mode=config['mode'], device=device,
                                       dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
                                       revision=config.get('model_revision', MODEL_REVISION))
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainables, lr=config['learning_rate'], weight_decay=config.get('weight_decay', .01))
    effective, epochs = config.get('effective_batch', 16), config.get('max_epochs', 20)
    max_steps = math.ceil(len(groups) / effective) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1. - step / max_steps))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    epoch, position, step, best, bad_epochs, sampler, history = 0, 0, 0, -1., 0, [], []
    epoch_loss_sum, epoch_loss_count = 0., 0
    if resume:
        # Checkpoints are local trusted pickle files. Never load an untrusted checkpoint.
        model.load_adapter_state_dict(saved['adapter'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        epoch, position, step = saved['epoch'], saved['sampler_position'], saved['global_step']
        best, bad_epochs = saved['best_metric'], saved['bad_epochs']
        sampler, history = saved['sampler'], saved['history']
        epoch_loss_sum, epoch_loss_count = saved['epoch_loss_sum'], saved['epoch_loss_count']
        set_rng(saved['rng'])
    atomic_json(result_dir / 'source_content_binding.json', source_binding)
    atomic_json(result_dir / 'admission.json', admission)
    atomic_json(result_dir / 'budget.json', dict(budget, validation_groups=len({r['group_id'] for r in val_rows}),
                                                validation_qa=len(val_rows),
                                                total_training_plus_validation_image_groups=budget['actual_groups'] + len({r['group_id'] for r in val_rows})))
    atomic_json(result_dir / 'parameter_report.json', model.parameter_report())
    started = time.monotonic()
    old_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in old_handlers:
        signal.signal(number, request_stop)

    def payload():
        return {'schema_version': 1, 'public_port_version': 1, 'adapter': model.adapter_state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(), 'sampler': sampler, 'sampler_position': position,
                'epoch': epoch, 'global_step': step, 'best_metric': best, 'bad_epochs': bad_epochs,
                'history': history, 'epoch_loss_sum': epoch_loss_sum, 'epoch_loss_count': epoch_loss_count,
                'data_hash': data_hash, 'source_content_hash': source_hash,
                'base_model_content_hash': object_hash(source_binding.get('base_model_files_sha256', {})),
                'config_hash': config_hash, 'training_code_hash': code_hash,
                'model_revision': config.get('model_revision', MODEL_REVISION), 'config': config, 'saved_at': now()}

    try:
        while epoch < epochs and bad_epochs < config.get('patience', 4):
            if not sampler:
                order = sorted(groups)
                random.shuffle(order)
                sampler = [random.choice(groups[group])['qa_id'] for group in order]
                position = 0
            lookup = {row['qa_id']: row for row in train_rows}
            model.train_mode()
            while position < len(sampler):
                if STOP or time.monotonic() - started > config.get('timeout_seconds', 10800):
                    save_checkpoint(latest, payload())
                    atomic_json(result_dir / 'run_status.json', {'status': 'RESUMABLE', 'global_step': step,
                                                               'reason': 'signal_or_runtime_limit', 'saved_at': now()})
                    return 75
                batch = [lookup[key] for key in sampler[position:position + effective]]
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.
                for row in batch:
                    loss = model.loss([image_for(row)], [row['question']], [row['answer']])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite supervised answer loss')
                    (loss / len(batch)).backward()
                    batch_loss += float(loss.detach()) / len(batch)
                gradient = torch.nn.utils.clip_grad_norm_(trainables, config.get('clip_norm', 1.))
                if not torch.isfinite(gradient) or float(gradient) == 0:
                    raise FloatingPointError('zero or nonfinite adapter gradient')
                optimizer.step()
                scheduler.step()
                position += len(batch)
                step += 1
                epoch_loss_sum += batch_loss
                epoch_loss_count += 1
                if step % config.get('checkpoint_interval', 10) == 0:
                    save_checkpoint(latest, payload())
                atomic_json(result_dir / 'heartbeat.json', {'epoch': epoch, 'global_step': step,
                                                          'sampler_position': position, 'loss': batch_loss,
                                                          'pid': os.getpid(), 'updated_at': now()})
            validation = result_dir / ('val_epoch_%02d.jsonl' % epoch)
            # An interrupted epoch may replay validation before its atomic commit.
            # Keep the earlier attempt instead of overwriting its predictions.
            if validation.exists():
                validation = validation.with_name(validation.stem + '_replay_%d.jsonl' % time.time_ns())
            metrics = evaluate_model(model, val_rows, validation, free_generation=False)['metrics']
            metric = metrics['type_balanced_group_accuracy']
            improved = metric > best
            best, bad_epochs = max(best, metric), 0 if improved else bad_epochs + 1
            history.append({'epoch': epoch, 'validation_metric': metric,
                            'train_loss': epoch_loss_sum / epoch_loss_count,
                            'training_optimizer_steps': epoch_loss_count})
            epoch += 1
            sampler, position, epoch_loss_sum, epoch_loss_count = [], 0, 0., 0
            if improved:
                save_checkpoint(best_path, {'schema_version': 1, 'adapter': model.adapter_state_dict(),
                                            'data_hash': data_hash, 'source_content_hash': source_hash,
                                            'base_model_content_hash': object_hash(source_binding.get('base_model_files_sha256', {})),
                                            'config_hash': config_hash, 'config': config,
                                            'selected_epoch': epoch - 1, 'global_step': step,
                                            'validation_metric': metric, 'training_code_hash': code_hash,
                                            'model_revision': config.get('model_revision', MODEL_REVISION)})
            save_checkpoint(latest, payload())
            atomic_json(result_dir / 'history.json', history)
        if not best_path.exists():
            raise RuntimeError('completed run has no committed best adapter')
        atomic_json(result_dir / 'run_status.json', {'status': 'SUCCESS', 'epochs': epoch, 'global_step': step,
                                                   'best_validation_metric': best, 'checkpoint': str(best_path),
                                                   'checkpoint_sha256': digest(best_path), 'updated_at': now()})
        return 0
    finally:
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def train(config_path, resume=False):
    config = load_config(config_path)
    out = Path(config['output_dir'])
    with run_lock(out / '.run.lock'):
        try:
            return _train(config, resume)
        except Exception as error:
            atomic_json(out / 'failed_attempts' / ('%d_%d.json' % (time.time_ns(), os.getpid())),
                        {'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error),
                         'config_hash': object_hash(config), 'updated_at': now()})
            raise
