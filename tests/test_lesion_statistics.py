"""Hand-computable confusion and sparse-support cases, not an efficacy dataset."""
import numpy as np
import pytest

from retinal_blip2.cli import score_lesions
from retinal_blip2.conditional_statistics import conditional_group_interval, conditional_outcomes
from retinal_blip2.lesion_statistics import calculate


def test_confusion_formula_with_invalid_predictions():
    metrics = calculate(np.array([3, 1, 2, 4, 5, 1]))
    assert metrics['positive_precision'] == pytest.approx(5/6)
    assert metrics['positive_recall'] == pytest.approx(5/10)
    assert metrics['positive_F1'] == pytest.approx(10/16)
    assert metrics['negative_F1'] == pytest.approx(6/13)
    assert metrics['accuracy'] == pytest.approx(8/16)


def test_unknown_reference_not_negative_and_sparse_interval_not_zero():
    row = dict(sample_id='synthetic',group_id='g',evidence_truth={key:False for key in ['EX','HE','MA','SE']},
               lesions={key:dict(prediction='no') for key in ['EX','HE','MA','SE']})
    report = score_lesions([row], 100)
    assert report['per_lesion']['EX']['metrics']['positive_F1']['point'] is None
    assert report['per_lesion']['EX']['metrics']['accuracy']['ci95'] is None
    row['evidence_truth']['EX'] = None
    with pytest.raises(ValueError, match='unknown evidence'):
        score_lesions([row],100)


def test_conditioned_success_is_not_population_accuracy():
    rows = [dict(group_id='a', evidence_correct=True, reasoning_correct=True),
            dict(group_id='b', evidence_correct=False, reasoning_correct=True),
            dict(group_id='c', evidence_correct=True, reasoning_correct=False),
            dict(group_id='d', evidence_correct=None, reasoning_correct=True)]
    result = conditional_outcomes(rows)
    assert result['conditional_answer_accuracy'] == .5
    assert result['answer_accuracy'] == .75
    assert result['excluded_unknown_required_evidence'] == 1
    assert result['four_cells'] == {'E_correct_R_correct':1,'E_correct_R_wrong':1,
                                    'E_wrong_R_correct':1,'E_wrong_R_wrong':0}
    assert conditional_group_interval(rows,n_bootstrap=100)['ci95'] is None
