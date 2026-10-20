"""Recompute outcomes from rows; patients/image groups are the sampling units."""
from collections import defaultdict, Counter
import numpy as np


def _bootstrap_count(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError('n_bootstrap must be a positive integer')


def _validate_predictions(rows):
    if not rows or len({r['qa_id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique QA predictions required')
    for row in rows:
        candidates = row['answer_candidates']
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError('answer candidates must be nonempty and unique')
        if row['answer'] not in candidates or row['prediction'] not in candidates:
            raise ValueError('truth/prediction outside frozen answer candidates')


def _group_type(rows, field='prediction'):
    values = defaultdict(list)
    for row in rows:
        values[(row['question_type'], row['group_id'])].append(
            float(row.get(field) == row['answer']))
    return {key: float(np.mean(value)) for key, value in values.items()}


def balanced_accuracy(rows, types=None, weights=None, field='prediction'):
    values = _group_type(rows, field)
    types = sorted(types or {r['question_type'] for r in rows})
    by_type = {}
    for kind in types:
        part = [(v, 1 if weights is None else weights.get(group, 0))
                for (t, group), v in values.items() if t == kind]
        denominator = sum(w for _, w in part)
        if denominator == 0:
            return None
        by_type[kind] = sum(v * w for v, w in part) / denominator
    return float(np.mean(list(by_type.values()))) if by_type else None


def summarize(rows, types=None):
    _validate_predictions(rows)
    types = sorted(types or {r['question_type'] for r in rows})
    if {r['question_type'] for r in rows} != set(types):
        raise ValueError('prediction question types differ from prespecified types')
    output = {'schema_version': 1, 'qa_rows': len(rows),
              'image_groups': len({r['group_id'] for r in rows}),
              'type_balanced_group_accuracy': balanced_accuracy(rows, types),
              'qa_weighted_accuracy': float(np.mean([r['prediction'] == r['answer'] for r in rows])),
              'per_type': {}}
    for kind in types:
        part = [r for r in rows if r['question_type'] == kind]
        if not part:
            raise ValueError('missing prespecified question type: ' + kind)
        classes = sorted({a for r in part for a in r['answer_candidates']})
        if any(r['prediction'] not in classes or r['answer'] not in classes for r in part):
            raise ValueError('truth/prediction outside frozen answer candidates')
        f1s = []
        for answer in classes:
            tp = sum(r['answer'] == answer and r['prediction'] == answer for r in part)
            fp = sum(r['answer'] != answer and r['prediction'] == answer for r in part)
            fn = sum(r['answer'] == answer and r['prediction'] != answer for r in part)
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.)
        confusion = Counter((r['answer'], r['prediction']) for r in part)
        output['per_type'][kind] = {
            'qa_rows': len(part), 'image_groups': len({r['group_id'] for r in part}),
            'group_accuracy': balanced_accuracy(part, [kind]),
            'qa_accuracy': float(np.mean([r['prediction'] == r['answer'] for r in part])),
            'macro_f1': float(np.mean(f1s)), 'macro_f1_classes': classes,
            'zero_support_f1_rule': 'zero; frozen candidates retained even if absent',
            'confusion': [{'truth': a, 'prediction': b, 'count': n}
                          for (a, b), n in sorted(confusion.items())]}
    generated = [r for r in rows if 'generation' in r]
    if generated:
        output['generation'] = {
            'qa_rows': len(generated),
            'invalid_fraction': float(np.mean([r.get('parsed_generation') is None for r in generated])),
            'strict_accuracy': float(np.mean([r.get('parsed_generation') == r['answer'] for r in generated]))}
    return output


def paired_bootstrap(method_a_seeds, method_b_seeds, types=None, n_bootstrap=2000, seed=271828):
    _bootstrap_count(n_bootstrap)
    if len(method_a_seeds) != len(method_b_seeds) or not method_a_seeds:
        raise ValueError('need aligned nonempty seed lists')
    first = method_a_seeds[0]
    identities = {r['qa_id']: (r['group_id'], r['question_type'], r['answer'],
                              tuple(r['answer_candidates'])) for r in first}
    if len(identities) != len(first):
        raise ValueError('duplicate QA IDs')
    for rows in method_a_seeds + method_b_seeds:
        _validate_predictions(rows)
        if {r['qa_id']: (r['group_id'], r['question_type'], r['answer'],
                         tuple(r['answer_candidates'])) for r in rows} != identities:
            raise ValueError('methods/seeds do not share identical QA/group/truth support')
    types = sorted(types or {r['question_type'] for r in first})
    if set(types) != {r['question_type'] for r in first}:
        raise ValueError('prediction question types differ from prespecified types')
    groups = sorted({r['group_id'] for r in first})
    arrays = []
    for seed_rows in (method_a_seeds, method_b_seeds):
        means = []
        for rows in seed_rows:
            values = _group_type(rows)
            means.append(np.array([[values.get((t, g), np.nan) for g in groups] for t in types]))
        arrays.append(np.stack(means))

    def score(array, counts):
        valid = ~np.isnan(array)
        denom = (valid * counts).sum(axis=2)
        if (denom == 0).any():
            return None
        numer = (np.nan_to_num(array) * counts).sum(axis=2)
        return float((numer / denom).mean())

    ones = np.ones(len(groups))
    point = score(arrays[0], ones) - score(arrays[1], ones)
    rng = np.random.RandomState(seed)
    draws, invalid, attempts = [], 0, 0
    # Fixed valid-resampling rule, bounded so rare types cannot hang a run.
    while len(draws) < n_bootstrap and attempts < n_bootstrap * 20:
        counts = np.bincount(rng.randint(0, len(groups), len(groups)), minlength=len(groups))
        attempts += 1
        a, b = score(arrays[0], counts), score(arrays[1], counts)
        if a is None or b is None:
            invalid += 1
            continue
        draws.append(a - b)
    if len(draws) != n_bootstrap:
        raise ValueError('insufficient valid group resamples: sparse prespecified types')
    return {'schema_version': 1, 'difference_a_minus_b': point,
            'ci95': np.percentile(draws, [2.5, 97.5]).tolist(),
            'n_bootstrap': len(draws), 'invalid_resamples': invalid,
            'groups': len(groups), 'fixed_models_per_method': len(method_a_seeds),
            'seed_effects': [balanced_accuracy(a, types) - balanced_accuracy(b, types)
                             for a, b in zip(method_a_seeds, method_b_seeds)],
            'interpretation': 'sample uncertainty conditional on fixed trained seeds; not training-randomness CI'}


def _pair_outcome(row):
    if row['kind'] not in ('answer_changed', 'answer_same'):
        raise ValueError('unknown image-pair kind')
    changed = row['left_truth'] != row['right_truth']
    if changed != (row['kind'] == 'answer_changed'):
        raise ValueError('image-pair kind disagrees with its two image truths')
    if len(row['groups']) != 2 or row['groups'][0] == row['groups'][1]:
        raise ValueError('image pair must contain two distinct groups')
    left = row['left_prediction'] == row['left_truth']
    right = row['right_prediction'] == row['right_truth']
    derived = {'joint_correct': left and right, 'mean_accuracy': (int(left) + int(right)) / 2.,
               'prediction_changed': row['left_prediction'] != row['right_prediction']}
    for field, value in derived.items():
        if field in row and row[field] != value:
            raise ValueError('saved image-pair outcome disagrees with raw predictions: ' + field)
    return derived


def pair_summary(rows):
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate image-pair predictions')
    outcomes = {r['pair_id']: _pair_outcome(r) for r in rows}
    result = {}
    for kind in ('answer_changed', 'answer_same'):
        part = [r for r in rows if r['kind'] == kind]
        result[kind] = {'pairs': len(part),
                        'joint_accuracy': float(np.mean([outcomes[r['pair_id']]['joint_correct'] for r in part])) if part else None,
                        'mean_accuracy': float(np.mean([outcomes[r['pair_id']]['mean_accuracy'] for r in part])) if part else None,
                        'prediction_change_rate': float(np.mean([outcomes[r['pair_id']]['prediction_changed'] for r in part])) if part else None}
    return result


def paired_joint_bootstrap(method_a_seeds, method_b_seeds, kind='answer_changed', n_bootstrap=2000, seed=271828):
    _bootstrap_count(n_bootstrap)
    if kind not in ('answer_changed', 'answer_same'):
        raise ValueError('unknown image-pair kind')
    if len(method_a_seeds) != len(method_b_seeds) or not method_a_seeds:
        raise ValueError('aligned nonempty seed predictions required')
    filtered = [[r for r in rows if r['kind'] == kind] for rows in method_a_seeds + method_b_seeds]
    reference = {r['pair_id']:(r['groups'],r['left_truth'],r['right_truth']) for r in filtered[0]}
    if not reference:
        return {'status':'NOT_ESTIMABLE','reason':'no eligible disjoint pairs','pairs':0}
    for rows in filtered:
        if len({r['pair_id'] for r in rows}) != len(rows) or len(rows) != len(reference) or {r['pair_id']:(r['groups'],r['left_truth'],r['right_truth']) for r in rows} != reference:
            raise ValueError('pair support/truth differs across seeds/methods')
    groups = [g for entry in reference.values() for g in entry[0]]
    if len(groups) != len(set(groups)):
        raise ValueError('pair bootstrap requires globally disjoint image groups')
    ids = sorted(reference)
    arrays=[]
    for rows in filtered:
        lookup={r['pair_id']:float(_pair_outcome(r)['joint_correct']) for r in rows}
        arrays.append([lookup[k] for k in ids])
    n_seeds=len(method_a_seeds)
    difference=np.asarray(arrays[:n_seeds]).mean(axis=0)-np.asarray(arrays[n_seeds:]).mean(axis=0)
    rng=np.random.RandomState(seed)
    effects=[float(difference[rng.randint(0,len(ids),len(ids))].mean()) for _ in range(n_bootstrap)]
    return {'schema_version':1,'difference_a_minus_b':float(difference.mean()),
            'ci95':np.percentile(effects,[2.5,97.5]).tolist(),'pairs':len(ids),'image_groups':len(groups),
            'n_bootstrap':n_bootstrap,'fixed_models_per_method':n_seeds,
            'sampling_unit':'disjoint_image_pair_cluster',
            'interpretation':'sample uncertainty conditional on fixed trained seeds'}
