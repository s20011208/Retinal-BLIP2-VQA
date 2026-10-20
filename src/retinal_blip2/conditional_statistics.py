"""Explicit denominators and paired original-group uncertainty for this study."""
from collections import Counter, defaultdict
import math
import numpy as np


def division(numerator,denominator):
    return numerator/denominator if denominator else None


def classification(truth,prediction,labels):
    if len(truth)!=len(prediction):raise ValueError('truth/prediction length differs')
    if any(t not in labels for t in truth):raise ValueError('unknown canonical label')
    columns=list(labels)+['INVALID']
    confusion=[[0 for _ in columns] for _ in labels]
    for t,p in zip(truth,prediction):
        confusion[labels.index(t)][columns.index(p) if p in labels else len(labels)]+=1
    per_class={}
    for index,label in enumerate(labels):
        tp=confusion[index][index]
        support=sum(confusion[index]);predicted=sum(row[index] for row in confusion)
        precision=division(tp,predicted);recall=division(tp,support)
        # F1 is defined when the class appears in truth or prediction, even if TP=0.
        f1=division(2*tp,support+predicted)
        per_class[label]=dict(support=support,predicted=predicted,true_positive=tp,
                              precision=precision,recall=recall,f1=f1,
                              truth_supported=support>0)
    supported=[v['f1'] for v in per_class.values() if v['support']>0]
    return dict(n=len(truth),correct=sum(t==p for t,p in zip(truth,prediction)),
        accuracy=division(sum(t==p for t,p in zip(truth,prediction)),len(truth)),
        macro_f1_over_truth_supported_classes=float(np.mean(supported)) if supported else None,
        missing_truth_classes=[label for label,v in per_class.items() if not v['support']],
        invalid_predictions=sum(p not in labels for p in prediction),
        per_class=per_class,confusion=confusion,truth_order=list(labels),prediction_order=columns)


def evidence_correct(truth,prediction,required):
    """Return unknown if any REQUIRED truth is absent; unknown is never correct."""
    if not required or len(set(required))!=len(required):raise ValueError('required evidence fields empty or duplicated')
    if any(key not in truth or truth[key] is None for key in required):return None
    if any(key not in prediction or prediction[key] is None for key in required):return False
    return all(prediction[key]==truth[key] for key in required)


def conditional_outcomes(rows,minimum_conditional_groups=20):
    """Rows contain explicitly adjudicated E correctness and final answer truth."""
    eligible=[r for r in rows if r.get('answer_truth_available',True)]
    excluded_unannotated_answer=len(rows)-len(eligible)
    known=[r for r in eligible if r['evidence_correct'] is not None]
    e_correct=[r for r in known if r['evidence_correct']]
    cells=Counter((bool(r['evidence_correct']),bool(r['reasoning_correct'])) for r in known)
    joint=cells[(True,True)]
    conditional=division(joint,len(e_correct))
    conditional_groups=len({r['group_id'] for r in e_correct})
    status=('NOT_ESTIMABLE' if conditional is None else
            'DESCRIPTIVE_ONLY_SPARSE_CONDITION' if conditional_groups<minimum_conditional_groups else 'ESTIMABLE_DESCRIPTIVE')
    return dict(total_rows=len(rows),answer_eligible=len(eligible),
        excluded_unannotated_answer=excluded_unannotated_answer,
        evidence_adjudicable=len(known),excluded_unknown_required_evidence=len(eligible)-len(known),
        answer_correct=sum(bool(r['reasoning_correct']) for r in eligible),
        answer_accuracy=division(sum(bool(r['reasoning_correct']) for r in eligible),len(eligible)),
        evidence_all_required_correct=len(e_correct),
        evidence_all_required_accuracy=division(len(e_correct),len(known)),
        conditional_answer_correct=joint,conditional_denominator=len(e_correct),
        conditional_original_groups=conditional_groups,
        conditional_answer_accuracy=conditional,
        conditional_status=status,minimum_conditional_original_groups=minimum_conditional_groups,
        joint_correct=joint,joint_denominator=len(known),joint_accuracy=division(joint,len(known)),
        four_cells={'E_correct_R_correct':cells[(True,True)],'E_correct_R_wrong':cells[(True,False)],
                    'E_wrong_R_correct':cells[(False,True)],'E_wrong_R_wrong':cells[(False,False)]},
        note='Answer correctness measures the declared endpoint, not validity of unobserved internal reasoning.')


def group_bootstrap(rows,statistic,n_bootstrap=2000,seed=20261020,support_filter=None,minimum_groups=20):
    """All rows from one original/patient group travel together; no zero fill."""
    by_group=defaultdict(list)
    for row in rows:by_group[row['group_id']].append(row)
    groups=sorted(by_group);point=statistic(rows)
    if point is None or not groups:
        return dict(status='NOT_ESTIMABLE',point=point,groups=len(groups),ci95=None,
                    reason='undefined denominator or no eligible original groups',n_bootstrap=n_bootstrap)
    effective_groups=len({r['group_id'] for r in rows if support_filter is None or support_filter(r)})
    if effective_groups<minimum_groups:
        return dict(status='NOT_ESTIMABLE_RELIABLY_SPARSE_SUPPORT',point=point,groups=len(groups),
                    effective_original_groups=effective_groups,minimum_original_groups=minimum_groups,
                    ci95=None,n_bootstrap=n_bootstrap,valid_resamples=0,invalid_resamples=0,
                    reason='descriptive point retained; actual estimand support below prespecified original-group threshold')
    rng=np.random.RandomState(seed);values=[];invalid=0
    for _ in range(n_bootstrap):
        sampled=rng.randint(0,len(groups),len(groups))
        resampled=[row for i in sampled for row in by_group[groups[i]]]
        value=statistic(resampled)
        if value is None or not math.isfinite(value):invalid+=1
        else:values.append(value)
    if len(values)<max(100,int(.8*n_bootstrap)):
        return dict(status='NOT_ESTIMABLE_RELIABLY',point=point,ci95=None,
                    valid_resamples=len(values),invalid_resamples=invalid,n_bootstrap=n_bootstrap,
                    reason='too many zero-denominator/undefined group resamples')
    return dict(status='ESTIMATED_WITH_SPARSE_SUPPORT_WARNING' if len(groups)<20 else 'ESTIMATED',
        point=point,ci95=[float(v) for v in np.percentile(values,[2.5,97.5])],
        groups=len(groups),effective_original_groups=effective_groups,valid_resamples=len(values),invalid_resamples=invalid,n_bootstrap=n_bootstrap,
        seed=seed,independence_unit='declared patient/verified-original group, not QA row')


def paired_group_difference(rows_a,rows_b,key,n_bootstrap=2000,seed=20261020):
    """Join the same sample identity before any bootstrap comparison."""
    a={r['sample_id']:r for r in rows_a};b={r['sample_id']:r for r in rows_b}
    if len(a)!=len(rows_a) or len(b)!=len(rows_b) or set(a)!=set(b):
        raise ValueError('paired comparison needs unique identical sample support')
    paired=[]
    for identity in sorted(a):
        if a[identity]['group_id']!=b[identity]['group_id']:raise ValueError('group identity mismatch')
        paired.append(dict(sample_id=identity,group_id=a[identity]['group_id'],a=a[identity],b=b[identity]))
    def difference(sample):
        left=conditional_outcomes([r['a'] for r in sample])[key]
        right=conditional_outcomes([r['b'] for r in sample])[key]
        return left-right if left is not None and right is not None else None
    counts=None;support_filter=None
    if key.startswith('conditional_'):
        counts={side:len({r['group_id'] for r in source if r['evidence_correct'] is True and r.get('answer_truth_available',True)})
                for side,source in [('a',rows_a),('b',rows_b)]}
        smaller=min(counts,key=counts.get)
        support_filter=lambda row:row[smaller]['evidence_correct'] is True and row[smaller].get('answer_truth_available',True)
    result=group_bootstrap(paired,difference,n_bootstrap,seed,support_filter=support_filter)
    if counts is not None:
        result['conditional_original_group_support_by_model']=counts
        result['conditional_comparison_boundary']='model-specific selected subsets; separate common-correct analysis required'
    return result


def conditional_group_interval(rows,n_bootstrap=2000,seed=20261020):
    """Required public entry for conditional CIs; support cannot be forgotten."""
    return group_bootstrap(rows,
        lambda sampled:conditional_outcomes(sampled)['conditional_answer_accuracy'],
        n_bootstrap=n_bootstrap,seed=seed,
        support_filter=lambda row:row['evidence_correct'] is True and row.get('answer_truth_available',True))


def common_correct_subset(by_model):
    lookups={model:{r['sample_id']:r for r in rows} for model,rows in by_model.items()}
    if any(len(lookups[model])!=len(rows) for model,rows in by_model.items()):
        raise ValueError('duplicate sample in common-subset comparison')
    supports=[set(v) for v in lookups.values()]
    if not supports or any(s!=supports[0] for s in supports):raise ValueError('common subset requires identical full support')
    for identity in supports[0]:
        if len({rows[identity]['group_id'] for rows in lookups.values()})!=1:
            raise ValueError('common subset group identity mismatch')
    common=sorted(i for i in supports[0] if all(rows[i]['evidence_correct'] is True for rows in lookups.values()))
    return dict(sample_ids=common,original_groups=len({lookups[next(iter(lookups))][i]['group_id'] for i in common}),
                models={model:conditional_outcomes([rows[i] for i in common]) for model,rows in lookups.items()},
                selection='all compared models recognize every required annotated evidence item correctly',
                note='Descriptive jointly selected subset; does not establish unselected population or causal reasoning superiority.')
