"""
Pytest test cases for ML Metrics Calculator
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ml_metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score


def test_accuracy_perfect_prediction():
    """Test accuracy with perfect predictions"""
    result = calculate_accuracy(100, 100, 0, 0)
    assert result == 1.0, "Perfect prediction should give 100% accuracy"


def test_accuracy_half_correct():
    """Test accuracy with 50% correct predictions"""
    result = calculate_accuracy(50, 50, 50, 50)
    assert result == 0.5, "Half correct should give 50% accuracy"


def test_precision_perfect():
    """Test precision with no false positives"""
    result = calculate_precision(100, 0)
    assert result == 1.0, "No false positives should give 100% precision"


def test_precision_half():
    """Test precision with equal true and false positives"""
    result = calculate_precision(50, 50)
    assert result == 0.5, "Equal TP and FP should give 50% precision"


def test_recall_perfect():
    """Test recall with no false negatives"""
    result = calculate_recall(100, 0)
    assert result == 1.0, "No false negatives should give 100% recall"


def test_recall_half():
    """Test recall with equal true positives and false negatives"""
    result = calculate_recall(50, 50)
    assert result == 0.5, "Equal TP and FN should give 50% recall"


def test_f1_score_perfect():
    """Test F1 score with perfect precision and recall"""
    result = calculate_f1_score(1.0, 1.0)
    assert result == 1.0, "Perfect precision and recall should give F1 of 1.0"


def test_f1_score_balanced():
    """Test F1 score with balanced precision and recall"""
    result = calculate_f1_score(0.8, 0.6)
    expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)
    assert abs(result - expected) < 0.001, f"F1 score should be {expected}"


def test_edge_case_zero_division():
    """Test edge cases that could cause division by zero"""
    assert calculate_accuracy(0, 0, 0, 0) == 0.0
    assert calculate_precision(0, 0) == 0.0
    assert calculate_recall(0, 0) == 0.0
    assert calculate_f1_score(0, 0) == 0.0