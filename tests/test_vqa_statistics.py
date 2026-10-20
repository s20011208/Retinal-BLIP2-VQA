import unittest
from retinal_blip2.statistics import balanced_accuracy, paired_bootstrap, summarize, pair_summary, paired_joint_bootstrap
from retinal_blip2.image_pairs import make_pairs, pair_results
from retinal_blip2.data import grouped_budget


def row(i, g, t, a, p):
    return dict(qa_id=i, group_id=g, canonical_image_id=g, question_type=t,
                question='Is it present?', answer=a, prediction=p, split='train',
                source_dataset='fixture_only', answer_candidates=['yes', 'no'],
                annotation_complete=True, question_roi_path_or_null=None)


class GroupStatisticsTests(unittest.TestCase):
    def test_qa_replication_cannot_inflate_group_metric(self):
        rows = [row('a'+str(i), 'g1', 'presence', 'yes', 'yes') for i in range(10)]
        rows += [row('b', 'g2', 'presence', 'no', 'yes'),
                 row('c', 'g3', 'grade', 'no', 'no')]
        self.assertEqual(balanced_accuracy(rows), .75)
        self.assertNotEqual(summarize(rows)['qa_weighted_accuracy'], .75)

    def test_bootstrap_pairs_all_seeds_and_groups(self):
        a = [row('a', 'g1', 't', 'yes', 'yes'), row('b', 'g2', 't', 'no', 'no')]
        b = [dict(r, prediction='yes' if r['answer'] == 'no' else 'no') for r in a]
        result = paired_bootstrap([a, a, a], [b, b, b], n_bootstrap=200)
        self.assertEqual(result['ci95'], [1., 1.])
        self.assertEqual(result['difference_a_minus_b'], 1.)
        with self.assertRaises(ValueError):
            paired_bootstrap([a], [[dict(b[0], answer='other'), b[1]]])

    def test_swap_keeps_new_truth_and_group_disjointness(self):
        rows = [row(str(i), 'g'+str(i), 't', 'yes' if i < 3 else 'no', 'yes') for i in range(6)]
        pairs, _ = make_pairs(rows)
        seen = []
        for pair in pairs:
            seen.extend([pair['left']['group_id'], pair['right']['group_id']])
            if pair['kind'] == 'answer_changed':
                self.assertNotEqual(pair['left']['answer'], pair['right']['answer'])
        self.assertEqual(len(seen), len(set(seen)))
        result = pair_results(pairs, rows)
        self.assertTrue(all(not r['joint_correct'] for r in result if r['kind'] == 'answer_changed'))

    def test_nested_budget_all_rows_inherit_group(self):
        rows = [row(str(i), 'g'+str(i//2), 't', 'yes' if i < 10 else 'no', 'yes') for i in range(20)]
        small, _ = grouped_budget(rows, .1, 17)
        large, _ = grouped_budget(rows, .25, 17)
        self.assertLessEqual({r['group_id'] for r in small}, {r['group_id'] for r in large})
        for group in {r['group_id'] for r in small}:
            self.assertEqual(sum(r['group_id'] == group for r in small), 2)

    def test_saved_pair_metrics_must_match_raw_predictions(self):
        pair = dict(pair_id='fixture', kind='answer_changed', groups=['a', 'b'],
                    left_truth='yes', right_truth='no', left_prediction='no', right_prediction='yes',
                    joint_correct=True, mean_accuracy=1., prediction_changed=False)
        with self.assertRaisesRegex(ValueError, 'raw predictions'):
            pair_summary([pair])
        with self.assertRaises(ValueError):
            paired_joint_bootstrap([[pair]], [[pair]], n_bootstrap=10)
        raw = {k: v for k, v in pair.items() if k not in ('joint_correct', 'mean_accuracy', 'prediction_changed')}
        self.assertEqual(pair_summary([raw])['answer_changed']['joint_accuracy'], 0.)
        self.assertEqual(paired_joint_bootstrap([[raw]], [[raw]], n_bootstrap=10)['difference_a_minus_b'], 0.)

    def test_bootstrap_missing_type_count_and_candidates_fail_explicitly(self):
        rows = [row('q', 'g', 't', 'yes', 'yes')]
        for options in ({'types': ['missing']}, {'n_bootstrap': 0}):
            with self.assertRaises(ValueError):
                paired_bootstrap([rows], [rows], **options)
        mutated = [dict(rows[0], answer_candidates=['yes', 'maybe'])]
        with self.assertRaises(ValueError):
            paired_bootstrap([rows], [mutated], n_bootstrap=10)


if __name__ == '__main__':
    unittest.main()
