import unittest
from scripts.benchmark import compute_evaluation_metrics

class TestBenchmarkMetrics(unittest.TestCase):
    def test_perfect_classification(self):
        metrics = compute_evaluation_metrics(tp=50, fp=0, tn=50, fn=0)
        self.assertEqual(metrics['accuracy'], 100.0)
        self.assertEqual(metrics['precision'], 100.0)
        self.assertEqual(metrics['recall'], 100.0)
        self.assertEqual(metrics['f1'], 100.0)

    def test_mixed_classification(self):
        # 80 TP, 20 FP, 70 TN, 30 FN
        # Total = 200, Acc = 150/200 = 75.0%
        # Precision = 80/100 = 80.0%
        # Recall = 80/110 = 72.73%
        metrics = compute_evaluation_metrics(tp=80, fp=20, tn=70, fn=30)
        self.assertEqual(metrics['accuracy'], 75.0)
        self.assertEqual(metrics['precision'], 80.0)
        self.assertAlmostEqual(metrics['recall'], 72.73, places=1)

    def test_empty_set_zero_division(self):
        metrics = compute_evaluation_metrics(tp=0, fp=0, tn=0, fn=0)
        self.assertEqual(metrics['accuracy'], 0.0)
        self.assertEqual(metrics['f1'], 0.0)

if __name__ == '__main__':
    unittest.main()
