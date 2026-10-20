"""Pair only audited, identical question semantics; each image keeps its truth."""
from collections import defaultdict
import random


def make_pairs(rows, seed=17):
    pools = defaultdict(list)
    excluded = 0
    for row in rows:
        if row.get('question_roi_path_or_null') or not row.get('annotation_complete'):
            excluded += 1
            continue
        key = (row['source_dataset'], row['question_type'], row['question'],
               tuple(row['answer_candidates']))
        pools[key].append(row)
    rng = random.Random(seed)
    # Assign each group one eligible semantic before pairing. Otherwise the
    # alphabetically first question type consumes every image and hides others.
    by_group = defaultdict(list)
    for key, pool in pools.items():
        for row in pool:
            by_group[row['group_id']].append((key, row))
    pools = defaultdict(list)
    for group in sorted(by_group):
        eligible = sorted(by_group[group], key=lambda item: item[1]['qa_id'])
        key, row = rng.choice(eligible)
        pools[key].append(row)
    used = set()
    pairs = []
    # Global non-overlap makes pair clusters independent image groups.
    for key in sorted(pools):
        pool = sorted(pools[key], key=lambda r: r['qa_id'])
        rng.shuffle(pool)
        # Reserve disjoint halves for changed and unchanged controls; a fully
        # balanced dataset must not lose the unchanged control to changed pairs.
        midpoint = (len(pool) + 1) // 2
        for changed, candidates in ((True, pool[:midpoint]), (False, pool[midpoint:])):
            for left in candidates:
                if left['group_id'] in used:
                    continue
                match = next((right for right in candidates
                              if right['group_id'] not in used
                              and right['group_id'] != left['group_id']
                              and (right['answer'] != left['answer']) == changed), None)
                if match is None:
                    continue
                used.update((left['group_id'], match['group_id']))
                pairs.append({'pair_id': 'pair_%06d' % len(pairs),
                              'kind': 'answer_changed' if changed else 'answer_same',
                              'left': left, 'right': match})
    return pairs, {'excluded_unverifiable_qa': excluded, 'pairs': len(pairs),
                   'paired_groups': len(used), 'pair_policy': 'globally_disjoint_groups'}


def pair_results(pairs, predictions):
    by_id = {r['qa_id']: r for r in predictions}
    output = []
    for pair in pairs:
        left, right = pair['left'], pair['right']
        a, b = by_id[left['qa_id']], by_id[right['qa_id']]
        output.append({'pair_id': pair['pair_id'], 'kind': pair['kind'],
                       'groups': [left['group_id'], right['group_id']],
                       'left_qa_id': left['qa_id'], 'right_qa_id': right['qa_id'],
                       'left_truth': left['answer'], 'right_truth': right['answer'],
                       'left_prediction': a['prediction'], 'right_prediction': b['prediction'],
                       'joint_correct': a['prediction'] == left['answer'] and
                                        b['prediction'] == right['answer'],
                       'mean_accuracy': ((a['prediction'] == left['answer']) +
                                         (b['prediction'] == right['answer'])) / 2,
                       'prediction_changed': a['prediction'] != b['prediction']})
    return output
