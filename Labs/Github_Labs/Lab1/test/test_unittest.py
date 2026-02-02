"""Unittest tests for ML Metrics"""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1

class TestMLMetrics(unittest.TestCase):
    
    def test_accuracy(self):
        self.assertEqual(calculate_accuracy(50, 50, 0, 0), 1.0)
        self.assertAlmostEqual(calculate_accuracy(50, 50, 25, 25), 0.667, places=2)
    
    def test_precision(self):
        self.assertEqual(calculate_precision(100, 0), 1.0)
        self.assertEqual(calculate_precision(50, 50), 0.5)
    
    def test_recall(self):
        self.assertEqual(calculate_recall(100, 0), 1.0)
        self.assertEqual(calculate_recall(50, 50), 0.5)
    
    def test_f1(self):
        self.assertEqual(calculate_f1(1.0, 1.0), 1.0)
        self.assertAlmostEqual(calculate_f1(0.8, 0.6), 0.686, places=2)

if __name__ == '__main__':
    unittest.main()