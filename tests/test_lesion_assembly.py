"""Guard the real QA-to-image metric boundary without relying on question text."""
import copy

import pytest

from retinal_blip2.lesions import assemble_lesions
from retinal_blip2.cli import score_lesions


MAPPING={'question_type_to_lesion':{'lesion_'+key:key for key in ['EX','HE','MA','SE']}}


def qa_rows():
    return [dict(qa_id='image::'+key,canonical_image_id='image',group_id='original_group',split='val',
                 question_type='lesion_'+key,question='Never parsed as a lesion name',answer_candidates=['no','yes'],
                 answer='yes' if key in ('EX','HE') else 'no',prediction='yes',annotation_complete=True)
            for key in ['EX','HE','MA','SE']]


def test_complete_join_and_metric_chain():
    joined=assemble_lesions(qa_rows(),MAPPING)
    assert len(joined)==1 and joined[0]['sample_id']=='image'
    assert joined[0]['evidence_truth']=={'EX':True,'HE':True,'MA':False,'SE':False}
    assert joined[0]['lesions']['EX']['question_type']=='lesion_EX'
    report=score_lesions(joined,100)
    assert report['per_lesion']['EX']['confusion']==[0,0,0,0,1,0]
    assert report['per_lesion']['MA']['confusion']==[0,1,0,0,0,0]


def test_missing_and_duplicate_channels_are_rejected():
    with pytest.raises(ValueError,match='all four'):
        assemble_lesions(qa_rows()[:-1],MAPPING)
    with pytest.raises(ValueError,match='duplicate QA'):
        assemble_lesions(qa_rows()+[qa_rows()[0]],MAPPING)
    duplicate=dict(qa_rows()[0],qa_id='another_EX')
    with pytest.raises(ValueError,match='duplicate lesion'):
        assemble_lesions(qa_rows()+[duplicate],MAPPING)


@pytest.mark.parametrize('field,value',[('answer','true'),('prediction','present'),('answer_candidates',['no','yes','maybe']),
                                       ('annotation_complete','true'),('lesion_type','SE'),('group_id','different')])
def test_ambiguous_or_conflicting_semantics_rejected(field,value):
    rows=qa_rows()
    rows[0][field]=value
    with pytest.raises(ValueError):
        assemble_lesions(rows,MAPPING)
