"""Consequential role guards reject forbidden answers before their JSON decoder."""
import json
from unittest.mock import patch

import pytest

from retinal_blip2 import data
from retinal_blip2.io import write_jsonl


def identity(key, role, group=None):
    return dict(qa_id=key, canonical_image_id=group or key, group_id=group or key,
                role=role, image_path=key+'.png', source_dataset='synthetic_test_only')


def answer(key, role):
    return dict(qa_id=key, role=role, question='fixture', question_type='fixture', answer_candidates=['no','yes'],
                answer='yes', annotation_complete=True, answer_provenance='synthetic_test_only')


def test_other_role_and_unadmitted_identity_are_not_decoded(tmp_path):
    ids, qa = tmp_path/'ids.jsonl', tmp_path/'qa.jsonl'
    write_jsonl(ids, [identity('allowed', 'train'), identity('final', 'test')])
    forbidden = '{"qa_id":"final", "role":"test", "answer":FORBIDDEN_NOT_VALID_JSON}\n'
    unknown = '{"qa_id":"other_project", "role":"train", "answer":FORBIDDEN_NOT_VALID_JSON}\n'
    qa.write_text(json.dumps(answer('allowed','train'))+'\n'+forbidden+unknown)
    actual_loads = json.loads
    def admitted_only(raw, *args, **kwargs):
        assert 'FORBIDDEN_NOT_VALID_JSON' not in raw
        return actual_loads(raw, *args, **kwargs)
    with patch.object(data.json, 'loads', side_effect=admitted_only):
        rows, audit = data.load_qa(ids, qa, ('train',), check_images=False)
    assert [row['qa_id'] for row in rows] == ['allowed']
    assert audit['excluded_before_answer_decode'] == 2
    with pytest.raises(PermissionError, match='receipt'):
        data.load_qa(ids, qa, ('test',), check_images=False)


def test_escaped_or_duplicate_role_rejected_before_decode(tmp_path):
    ids, qa = tmp_path/'ids.jsonl', tmp_path/'qa.jsonl'
    write_jsonl(ids, [identity('allowed','train')])
    for raw in ('{"qa_id":"allowed","r\\u006fle":"train","answer":NEVER_DECODE}',
                '{"qa_id":"allowed","role":"train","role":"test","answer":NEVER_DECODE}'):
        qa.write_text(raw+'\n')
        with pytest.raises(PermissionError):
            data.load_qa(ids, qa, ('train',), check_images=False)


@pytest.mark.parametrize('field', ['group_id','canonical_image_id','patient_id','raw_sha256','pixel_sha256'])
def test_known_identity_overlap_rejected(tmp_path, field):
    rows = [identity('a','train'), identity('b','val')]
    for row in rows:
        row[field] = 'same_identity'
    path = tmp_path/'ids.jsonl'
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match='cross-role'):
        data.identities(path, check_images=False)


def test_group_budget_never_selects_qa_independently():
    rows = [dict(qa_id='%d_%d'%(group,qa), group_id=str(group), split='train', question_type='presence',
                 canonical_image_id=str(group), answer=['no','yes'][group%2]) for group in range(12) for qa in range(3)]
    small, a = data.grouped_budget(rows,.1,17)
    big, b = data.grouped_budget(rows,.5,17)
    assert {row['qa_id'] for row in small} <= {row['qa_id'] for row in big}
    assert len(small) == 3*a['actual_groups']
    assert {row['answer'] for row in small} == {'yes','no'}


def test_string_false_is_not_complete_annotation(tmp_path):
    ids,qa=tmp_path/'ids.jsonl',tmp_path/'qa.jsonl'
    write_jsonl(ids,[identity('a','train')])
    reference=answer('a','train')
    reference['annotation_complete']='false'
    write_jsonl(qa,[reference])
    with pytest.raises(ValueError,match='incomplete'):
        data.load_qa(ids,qa,('train',),check_images=False)


def test_actual_pixel_duplicate_across_roles_rejected_without_supplied_hash(tmp_path):
    from PIL import Image
    picture=Image.new('RGB',(8,8),(42,17,61))
    picture.save(tmp_path/'a.png')
    picture.save(tmp_path/'b.bmp')
    rows=[identity('a','train'),identity('b','val')]
    rows[1]['image_path']='b.bmp'
    path=tmp_path/'ids.jsonl'
    write_jsonl(path,rows)
    with pytest.raises(ValueError,match='actual decoded RGB'):
        data.identities(path,data_root=tmp_path)


@pytest.mark.parametrize('field',['role','qa_id'])
@pytest.mark.parametrize('extra',['null','42','true','{}','[]'])
def test_nonstring_duplicate_markers_rejected_before_full_answer_decode(tmp_path,field,extra):
    ids,qa=tmp_path/'ids.jsonl',tmp_path/'qa.jsonl'
    write_jsonl(ids,[identity('allowed','train')])
    raw='{"qa_id":"allowed","role":"train","%s":%s,"answer":"SENTINEL_NEVER_DECODE"}'%(field,extra)
    qa.write_text(raw+'\n')
    actual=json.loads
    def never_full_decode(value,*args,**kwargs):
        assert 'SENTINEL_NEVER_DECODE' not in value
        return actual(value,*args,**kwargs)
    with patch.object(data.json,'loads',side_effect=never_full_decode):
        with pytest.raises(PermissionError,match='duplicate literal'):
            data.load_qa(ids,qa,('train',),check_images=False)


def test_final_role_requires_real_image_hash_binding_before_answer_decode(tmp_path):
    from PIL import Image
    from retinal_blip2.io import atomic_json,digest
    Image.new('RGB',(8,8),(10,20,30)).save(tmp_path/'final.png')
    ids,qa,receipt=tmp_path/'ids.jsonl',tmp_path/'qa.jsonl',tmp_path/'receipt.json'
    write_jsonl(ids,[identity('final','test')])
    qa.write_text('{"qa_id":"final","role":"test","answer":NEVER_DECODE}\n')
    atomic_json(receipt,dict(status='FROZEN_FOR_EVALUATION',identities_sha256=digest(ids),answers_sha256=digest(qa)))
    with pytest.raises(PermissionError,match='raw_sha256'):
        data.load_qa(ids,qa,('test',),data_root=tmp_path,receipt_path=receipt)
