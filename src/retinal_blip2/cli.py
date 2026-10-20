"""Public entry points. Data, adapters and model weights remain local."""
import argparse
import json
from pathlib import Path

from .data import load_qa
from .io import atomic_json, digest, load_config, object_hash, read_jsonl, write_jsonl
from .lesion_statistics import LESIONS, lesion_intervals
from .statistics import paired_bootstrap, paired_joint_bootstrap, pair_summary, summarize


def score_lesions(rows, n_bootstrap=2000, seed=20261020):
    """Image rows with explicit known truth and no implicit mask negatives."""
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('nonempty unique image-level lesion predictions required')
    for row in rows:
        if not row.get('group_id') or set(row.get('evidence_truth', {})) != set(LESIONS):
            raise ValueError('group identity and all four known lesion references required')
        if any(type(row['evidence_truth'][key]) not in (bool, int) or row['evidence_truth'][key] not in (0, 1) for key in LESIONS):
            raise ValueError('unknown evidence is not a lesion-negative reference')
        if set(row.get('lesions', {})) != set(LESIONS):
            raise ValueError('all four lesion prediction records required')
        if any(row['lesions'][key].get('prediction') not in ('yes', 'no', None) for key in LESIONS):
            raise ValueError('predictions must be yes/no/null; null remains invalid')
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError('at least 100 group bootstrap draws required')
    return lesion_intervals(rows, n_bootstrap=n_bootstrap, seed=seed)


def predict(config_path):
    from .evaluate import evaluate_model
    from .models.blip2_adapter import MODEL_REVISION, RetinalBLIP2
    import torch
    config = load_config(config_path)
    roles = config.get('roles', ['val'])
    if len(roles) != 1:
        raise ValueError('one explicit evaluation role per prediction artifact')
    mode = config.get('mode', 'B0')
    if mode not in ('B0', 'B1', 'M1'):
        raise ValueError('mode must be B0, B1 or M1')
    checkpoint = config.get('checkpoint')
    if mode != 'B0' and not checkpoint:
        raise ValueError('adapted-model evaluation requires its trained local checkpoint')
    if mode == 'B0' and checkpoint:
        raise ValueError('B0 must use the original model without adapter replacement')
    from .train import bind_model_sources
    base_content_hash = object_hash(bind_model_sources(config['model_path']))
    if 'test' in roles:
        if not config.get('receipt'):
            raise PermissionError('test prediction requires a frozen-study receipt before answer admission')
        receipt = json.loads(Path(config['receipt']).read_text())
        bound_config = {key: value for key, value in config.items() if key != 'receipt'}
        if (receipt.get('evaluation_config_sha256') != object_hash(bound_config) or
                receipt.get('checkpoint_sha256') != (digest(checkpoint) if checkpoint else None)):
            raise PermissionError('test receipt does not bind the exact resolved evaluation config and adapter')
        from .train import training_code_hash
        if receipt.get('code_sha256') != training_code_hash():
            raise PermissionError('test receipt does not bind the evaluation implementation')
        if receipt.get('base_model_content_hash') != base_content_hash:
            raise PermissionError('test receipt does not bind the actual base-model bytes')
    rows, admission = load_qa(config['identities'], config['answers'], roles,
                              config.get('data_root'), receipt_path=config.get('receipt'))
    device = config.get('device', 'cuda')
    model = RetinalBLIP2.from_pretrained(config['model_path'], mode=mode, device=device,
                                       dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
                                       revision=config.get('model_revision', MODEL_REVISION))
    if checkpoint:
        # Only local trusted checkpoints are supported; torch pickle may execute code.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        expected_revision = config.get('model_revision', MODEL_REVISION)
        if saved.get('model_revision') not in (None, expected_revision):
            raise ValueError('adapter base-model revision differs from configured revision')
        if (saved.get('base_model_content_hash') is not None and
                saved['base_model_content_hash'] != base_content_hash):
            raise ValueError('adapter was trained against different base-model file bytes')
        model.load_adapter_state_dict(saved['adapter'])
    report = evaluate_model(model, rows, config['output'], config.get('free_generation', True))
    report.update(admission=admission, mode=mode, model_revision=config.get('model_revision', MODEL_REVISION),
                  base_model_content_hash=base_content_hash,
                  adapter_binding=('base_content_verified' if checkpoint and saved.get('base_model_content_hash') else
                                   'historical_keys_shapes_declared_revision_only' if checkpoint else 'unadapted_base'),
                  checkpoint_sha256=digest(checkpoint) if checkpoint else None)
    atomic_json(Path(config['output']).with_suffix('.execution.json'), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='Check identity groups and admit role-permitted QA references')
    audit.add_argument('--identities', required=True)
    audit.add_argument('--answers', required=True)
    audit.add_argument('--roles', nargs='+', choices=('train', 'val'), default=['train', 'val'])
    audit.add_argument('--data-root', default='.')
    audit.add_argument('--output', required=True)
    training = sub.add_parser('train', help='Train B1/M1 with full-state interruption recovery')
    training.add_argument('--config', required=True)
    training.add_argument('--resume', action='store_true')
    inference = sub.add_parser('predict', help='Score frozen candidates and save strict free generation')
    inference.add_argument('--config', required=True)
    assemble = sub.add_parser('assemble-lesions', help='Join four explicitly typed lesion QAs per original image')
    assemble.add_argument('--predictions', required=True)
    assemble.add_argument('--mapping', required=True)
    assemble.add_argument('--output', required=True)
    for name, help_text in [('score-vqa', 'Recompute grouped/type-balanced VQA metrics from local predictions'),
                            ('score-lesions', 'Recompute four-lesion confusion metrics and grouped intervals'),
                            ('score-pairs', 'Recompute answer-changed/same image-pair metrics')]:
        scoring = sub.add_parser(name, help=help_text)
        scoring.add_argument('--predictions', required=True)
        scoring.add_argument('--output', required=True)
        if name == 'score-lesions':
            scoring.add_argument('--bootstrap', type=int, default=2000)
            scoring.add_argument('--seed', type=int, default=20261020)
    paired = sub.add_parser('compare-vqa', help='Paired original-group intervals; equally many fixed seed files per method')
    paired.add_argument('--a', nargs='+', required=True)
    paired.add_argument('--b', nargs='+', required=True)
    paired.add_argument('--output', required=True)
    paired.add_argument('--bootstrap', type=int, default=2000)
    paired.add_argument('--seed', type=int, default=271828)
    args = parser.parse_args(argv)
    if args.command == 'train':
        from .train import train
        return train(args.config, args.resume)
    if args.command == 'audit':
        _, report = load_qa(args.identities, args.answers, args.roles, args.data_root)
        report['status'] = 'PASS_DECLARED_IDENTITY_AND_ROLE_CHECKS'
    elif args.command == 'predict':
        report = predict(args.config)
        print(json.dumps({'status': 'COMPLETE', 'rows': report['rows'], 'mode': report['mode']}))
        return 0
    elif args.command == 'assemble-lesions':
        from .lesions import assemble_lesions
        rows = assemble_lesions(read_jsonl(args.predictions), load_config(args.mapping))
        write_jsonl(args.output, rows)
        print(json.dumps({'status':'COMPLETE','images':len(rows),'output':args.output}))
        return 0
    elif args.command == 'score-vqa':
        report = summarize(read_jsonl(args.predictions))
    elif args.command == 'score-lesions':
        report = score_lesions(read_jsonl(args.predictions), args.bootstrap, args.seed)
    elif args.command == 'score-pairs':
        report = pair_summary(read_jsonl(args.predictions))
    else:
        report = paired_bootstrap([read_jsonl(path) for path in args.a],
                                  [read_jsonl(path) for path in args.b],
                                  n_bootstrap=args.bootstrap, seed=args.seed)
    atomic_json(args.output, report)
    print(json.dumps({'status': 'COMPLETE', 'command': args.command, 'output': args.output}))
    return 0
