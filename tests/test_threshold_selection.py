import unittest

from fraudGT.evaluation.threshold_selection import (
    evaluate_thresholds,
    select_on_validation,
)


class ThresholdSelectionTest(unittest.TestCase):
    def test_metrics_and_selection(self):
        labels = [0, 0, 1, 1]
        scores = [0.05, 0.40, 0.30, 0.90]
        rows = evaluate_thresholds(labels, scores, [0.10, 0.50])

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['f1'], 0.8)
        self.assertAlmostEqual(rows[1]['f1'], 2 / 3)
        self.assertEqual(
            select_on_validation(labels, scores, [0.10, 0.50]),
            rows[0],
        )

    def test_tie_prefers_higher_threshold(self):
        selected = select_on_validation(
            [0, 1], [0.10, 0.90], [0.20, 0.50])
        self.assertEqual(selected['threshold'], 0.50)


if __name__ == '__main__':
    unittest.main()

