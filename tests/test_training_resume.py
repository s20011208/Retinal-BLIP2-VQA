"""Interrupt/continue the real optimizer loop with a tiny CPU loss fixture."""
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from retinal_blip2 import train as trainer
from retinal_blip2.io import atomic_json


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.frozen_lm = torch.nn.Parameter(torch.tensor(.5), requires_grad=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def train_mode(self):
        self.train()

    def loss(self, images, questions, answers):
        noise = torch.rand(())*.05 + random.random()*.05 + float(np.random.rand())*.05
        return (self.weight*float(images[0])*self.frozen_lm + noise - 1.).square()

    def adapter_state_dict(self):
        return {'weight':self.weight.detach().clone()}

    def load_adapter_state_dict(self, state):
        with torch.no_grad():
            self.weight.copy_(state['weight'])

    def parameter_report(self):
        return {'fixture_only':True,'trainable_parameters':1}


def same_state(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:
            same_state(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b):
            same_state(left,right)
    else:
        assert a==b


@pytest.fixture
def loop(tmp_path):
    rows = [dict(qa_id='q%d'%i, group_id='g%d'%i, canonical_image_id='g%d'%i,
                 question_type='presence', question='fixture only',answer='yes',feature=i+1,
                 split='train' if i<4 else 'val') for i in range(6)]
    def evaluate(model, rows, path, **kwargs):
        return {'metrics':{'type_balanced_group_accuracy':float(model.weight.detach())}}
    with patch.object(trainer,'RetinalBLIP2',TinyAdapter), \
            patch.object(trainer,'load_qa',return_value=(rows,dict(identities_sha256='fixture_ids',answers_sha256='fixture_qa'))), \
            patch.object(trainer,'image_for',side_effect=lambda row:float(row['feature'])), \
            patch.object(trainer,'evaluate_model',side_effect=evaluate), \
            patch.object(trainer,'training_code_hash',return_value='fixture_implementation'), \
            patch.object(trainer,'bind_training_sources',return_value={'fixture_binding':'same_actual_fixture_inputs'}):
        yield tmp_path


def config(root, run_id, epochs=2):
    value=dict(output_dir=str(root/run_id),seed=17,mode='B1',device='cpu',identities='fixture_ids',
               answers='fixture_qa',model_path='synthetic_cpu_fixture_only',budget=1.,learning_rate=.01,
               effective_batch=2,max_epochs=epochs,checkpoint_interval=1,patience=4,timeout_seconds=30)
    path=root/(run_id+'.json')
    atomic_json(path,value)
    return path


def test_interrupted_and_continuous_optimizer_and_rng_are_identical(loop):
    continuous, interrupted = config(loop,'continuous'),config(loop,'interrupted')
    assert trainer.train(continuous)==0
    original=trainer.save_checkpoint
    did_interrupt=[False]
    def save_then_stop(path,payload):
        original(path,payload)
        if Path(path).name=='latest.pt' and payload['epoch']==0 and payload['global_step']==1 and not did_interrupt[0]:
            trainer.STOP=True
            did_interrupt[0]=True
    with patch.object(trainer,'save_checkpoint',side_effect=save_then_stop):
        assert trainer.train(interrupted)==75
    assert trainer.train(interrupted,resume=True)==0
    a,b=[torch.load(loop/run/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
         for run in ('continuous','interrupted')]
    for field in ('adapter','optimizer','scheduler','scaler','rng','history','sampler','sampler_position',
                  'epoch','global_step','best_metric','bad_epochs'):
        same_state(a[field],b[field])


def test_selected_adapter_is_committed_before_epoch_latest(loop):
    path=config(loop,'commit_crash',1)
    original=trainer.save_checkpoint
    def save_then_crash(target,payload):
        original(target,payload)
        if Path(target).name=='latest.pt' and payload['epoch']==1:
            raise RuntimeError('fixture crash just after epoch commit')
    with patch.object(trainer,'save_checkpoint',side_effect=save_then_crash):
        with pytest.raises(RuntimeError,match='fixture crash'):
            trainer.train(path)
    assert (loop/'commit_crash/checkpoints/best.pt').is_file()
    assert trainer.train(path,resume=True)==0
    assert list((loop/'commit_crash/failed_attempts').glob('*.json'))


def test_resume_rejects_code_change_and_preserves_old_result(loop):
    path=config(loop,'changed',1)
    assert trainer.train(path)==0
    preserved={name:(loop/'changed/results'/name).read_bytes() for name in ['run_status.json','admission.json','budget.json']}
    with patch.object(trainer,'training_code_hash',return_value='different_implementation'):
        with pytest.raises(ValueError,match='implementation changed'):
            trainer.train(path,resume=True)
    assert {name:(loop/'changed/results'/name).read_bytes() for name in preserved}==preserved
    assert list((loop/'changed/failed_attempts').glob('*.json'))
    with pytest.raises(FileExistsError):
        trainer.train(path)


def test_invalid_resume_data_does_not_replace_old_provenance(loop):
    path=config(loop,'wrong_data',1)
    assert trainer.train(path)==0
    names=['admission.json','budget.json','parameter_report.json','run_status.json']
    preserved={name:(loop/'wrong_data/results'/name).read_bytes() for name in names}
    admitted,_=trainer.load_qa('fixture_ids','fixture_qa')
    with patch.object(trainer,'load_qa',return_value=(admitted,dict(identities_sha256='different',answers_sha256='different'))):
        with pytest.raises(ValueError):
            trainer.train(path,resume=True)
    assert {name:(loop/'wrong_data/results'/name).read_bytes() for name in names}==preserved


def test_only_selected_adapter_directory_is_never_overwritten(loop):
    path=config(loop,'only_best',1)
    assert trainer.train(path)==0
    latest=loop/'only_best/checkpoints/latest.pt'
    best=loop/'only_best/checkpoints/best.pt'
    before=best.read_bytes()
    latest.unlink()
    with pytest.raises(FileExistsError,match='selected adapter exists'):
        trainer.train(path)
    assert best.read_bytes()==before


def test_resume_rejects_same_paths_with_changed_image_or_model_bytes(loop):
    path=config(loop,'changed_bytes',1)
    assert trainer.train(path)==0
    names=['admission.json','budget.json','source_content_binding.json','run_status.json']
    preserved={name:(loop/'changed_bytes/results'/name).read_bytes() for name in names}
    with patch.object(trainer,'bind_training_sources',return_value={'fixture_binding':'different_image_bytes'}):
        with pytest.raises(ValueError,match='bytes changed'):
            trainer.train(path,resume=True)
    assert {name:(loop/'changed_bytes/results'/name).read_bytes() for name in names}==preserved


def test_source_binding_hashes_actual_bytes_even_if_path_and_mtime_preserved(tmp_path):
    image=tmp_path/'image.png'
    image.write_bytes(b'synthetic_image_bytes_A')
    model=tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'model.safetensors').write_bytes(b'synthetic_model_bytes_A')
    rows=[{'image_path':str(image)}]
    before=trainer.bind_training_sources(rows,model)
    image.write_bytes(b'synthetic_image_bytes_B')
    after=trainer.bind_training_sources(rows,model)
    assert before['image_files_sha256']!=after['image_files_sha256']
    assert before['base_model_files_sha256']==after['base_model_files_sha256']
    (model/'model.safetensors').write_bytes(b'synthetic_model_bytes_B')
    changed=trainer.bind_training_sources(rows,model)
    assert after['base_model_files_sha256']!=changed['base_model_files_sha256']
