"""ML Metrics Calculator - Santhosh K Chandrasekar"""

def calculate_accuracy(tp, tn, fp, fn):
    """Calculate accuracy"""
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0

def calculate_precision(tp, fp):
    """Calculate precision"""
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calculate_recall(tp, fn):
    """Calculate recall"""
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calculate_f1(precision, recall):
    """Calculate F1 score"""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0