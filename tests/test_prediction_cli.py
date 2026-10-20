"""Exercise the actual CLI orchestration with tiny local architecture fixtures."""
import json
from unittest.mock import patch

import pytest
from PIL import Image

from conftest import tiny_adapter
from retinal_blip2 import cli
from retinal_blip2.io import atomic_json, read_jsonl, write_jsonl
from retinal_blip2.models.blip2_adapter import RetinalBLIP2


def test_real_public_prediction_and_scoring_entry_points(tmp_path):
    image=tmp_path/'image.png'
    Image.new('RGB',(20,20),(20,70,40)).save(image)
    ids=tmp_path/'identities.jsonl'
    answers=tmp_path/'qa.jsonl'
    write_jsonl(ids,[dict(qa_id='q',canonical_image_id='original',group_id='original',role='val',
                          image_path=str(image),source_dataset='synthetic_test_only')])
    write_jsonl(answers,[dict(qa_id='q',role='val',question='Are there exudates?',question_type='lesion_EX',
                              answer_candidates=['no','yes'],answer='yes',annotation_complete=True,
                              answer_provenance='synthetic_test_only',lesion_type='EX')])
    output=tmp_path/'prediction.jsonl'
    config=tmp_path/'predict.json'
    atomic_json(config,dict(identities=str(ids),answers=str(answers),model_path='tiny_architecture_fixture',
                            output=str(output),mode='B0',device='cpu',roles=['val'],free_generation=True))
    model=tiny_adapter('B0')
    with patch.object(RetinalBLIP2,'from_pretrained',return_value=model), \
            patch('retinal_blip2.train.bind_model_sources',return_value={'synthetic_base':'not_research_weights'}):
        assert cli.main(['predict','--config',str(config)])==0
    rows=read_jsonl(output)
    assert len(rows)==1 and rows[0]['lesion_type']=='EX' and rows[0]['annotation_complete'] is True
    assert rows[0]['prediction'] in ['no','yes'] and 'generation' in rows[0]
    recalculated=tmp_path/'rescored.json'
    assert cli.main(['score-vqa','--predictions',str(output),'--output',str(recalculated)])==0
    assert json.loads(recalculated.read_text())==json.loads(output.with_suffix('.metrics.json').read_text())


def test_final_bindings_rejected_before_answer_loader(tmp_path):
    config=tmp_path/'final.json'
    atomic_json(config,dict(identities='never_read',answers='never_read',model_path='never_load',
                            mode='B0',roles=['test'],output='never_write'))
    with patch('retinal_blip2.train.bind_model_sources',return_value={'synthetic_base':'fixture'}), \
            patch.object(cli,'load_qa') as loader:
        with pytest.raises(PermissionError,match='receipt'):
            cli.predict(config)
        loader.assert_not_called()
