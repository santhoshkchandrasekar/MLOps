"""
ML Metrics Calculator
Calculates common machine learning evaluation metrics
Author: Santhosh K Chandrasekar
"""

def calculate_accuracy(true_positives, true_negatives, false_positives, false_negatives):
    """
    Calculate accuracy from confusion matrix components
    Formula: (TP + TN) / (TP + TN + FP + FN)
    """
    total = true_positives + true_negatives + false_positives + false_negatives
    if total == 0:
        return 0.0
    return (true_positives + true_negatives) / total


def calculate_precision(true_positives, false_positives):
    """
    Calculate precision (positive predictive value)
    Formula: TP / (TP + FP)
    """
    if (true_positives + false_positives) == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def calculate_recall(true_positives, false_negatives):
    """
    Calculate recall (sensitivity, true positive rate)
    Formula: TP / (TP + FN)
    """
    if (true_positives + false_negatives) == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score (harmonic mean of precision and recall)
    Formula: 2 * (precision * recall) / (precision + recall)
    """
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)