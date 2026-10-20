"""Actual HF tiny/random architecture tests. These are not medical experiments."""
import math

import pytest
import torch

from conftest import synthetic_samples, tiny_adapter
from retinal_blip2.models.answer_scoring import answer_log_scores, parse_answer


@pytest.mark.parametrize('expanded', [False, True])
@pytest.mark.parametrize('mode', ['B1', 'M1'])
def test_official_forward_gradient_freezing_scoring_and_roundtrip(expanded, mode, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = tiny_adapter(mode, expanded)
    images, questions, answers = synthetic_samples()
    model.eval()
    batch = model.build_batch(images, questions, answers)
    with torch.no_grad():
        official = model.model(**batch, return_dict=True)
        manual = model.manual_forward_logits(batch)[:, -batch['labels'].shape[1]:]
        assert torch.allclose(official.logits, manual, atol=1e-5, rtol=1e-5)
        scores = answer_log_scores(official.logits, batch['labels'])
        independent = 0.
        for index in torch.where(batch['labels'][0, 1:].ne(-100))[0].tolist():
            values = official.logits[0, index].float().tolist()
            token = int(batch['labels'][0, index + 1])
            largest = max(values)
            independent += values[token] - largest - math.log(sum(math.exp(item-largest) for item in values))
        assert abs(independent - scores[0].item()) < 2e-4
        expected_loss = -scores.sum() / batch['labels'][:, 1:].ne(-100).sum()
        assert torch.allclose(official.loss, expected_loss, atol=1e-5)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    before = model.adapter_state_dict()
    model.train()
    assert not model.model.vision_model.training and not model.model.language_model.training
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    optimizer.zero_grad()
    model.loss(images, questions, answers).backward()
    gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None}
    assert any('language_projection.' in name and value > 0 for name, value in gradients.items())
    if mode == 'M1':
        assert any('qformer.' in name and value > 0 for name, value in gradients.items())
        assert any('query_tokens' in name and value > 0 for name, value in gradients.items())
    else:
        assert not model.model.qformer.training
        assert all('language_projection.' in name for name in gradients)
    assert all(parameter.grad is None for parameter in model.model.language_model.parameters())
    assert all(parameter.grad is None for parameter in model.model.vision_model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.adapter_state_dict().items())
    assert all(torch.equal(frozen[name], parameter) for name, parameter in model.named_parameters() if name in frozen)
    scores = model.score_candidates(images, questions, [['no','yes']]*2)
    model.save_adapter(tmp_path/'adapter.pt')
    saved = torch.load(tmp_path/'adapter.pt', map_location='cpu', weights_only=False)
    with torch.no_grad():
        for parameter in model.model.language_projection.parameters():
            parameter.add_(100.)
    model.load_adapter_state_dict(saved['adapter'])
    assert scores == model.score_candidates(images, questions, [['no','yes']]*2)
    with pytest.raises(ValueError, match='keys mismatch'):
        model.load_adapter_state_dict({})


def test_termination_and_strict_whole_output():
    model = tiny_adapter()
    model.model.language_model.generation_config.eos_token_id = 14
    images, questions, answers = synthetic_samples()
    batch = model.build_batch(images[:1], questions[:1], answers[:1])
    assert batch['labels'][0, -1].item() == 14
    assert model.processor.tokenizer.eos_token_id == 2
    assert parse_answer(' Yes. ', ['no','yes']) == 'yes'
    assert parse_answer('yes, and severe hemorrhage', ['no','yes']) is None
    assert parse_answer('', ['no','yes']) is None
