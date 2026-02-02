"""Pytest tests for ML Metrics"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1

def test_accuracy():
    assert calculate_accuracy(50, 50, 0, 0) == 1.0
    assert calculate_accuracy(50, 50, 25, 25) == 0.667 or abs(calculate_accuracy(50, 50, 25, 25) - 0.667) < 0.01

def test_precision():
    assert calculate_precision(100, 0) == 1.0
    assert calculate_precision(50, 50) == 0.5

def test_recall():
    assert calculate_recall(100, 0) == 1.0
    assert calculate_recall(50, 50) == 0.5

def test_f1():
    assert calculate_f1(1.0, 1.0) == 1.0
    assert abs(calculate_f1(0.8, 0.6) - 0.686) < 0.01