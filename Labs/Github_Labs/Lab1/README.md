 # Lab 1 - ML Metrics Calculator

**Author:** Santhosh K Chandrasekar  
**Course:** DADS7305 MLOps  
**Modification:** Replaced basic calculator with ML evaluation metrics

## Overview

This lab implements an ML metrics calculator instead of a basic arithmetic calculator, demonstrating MLOps principles with a data science focus.

## Functions

- `calculate_accuracy(tp, tn, fp, fn)` - Model accuracy
- `calculate_precision(tp, fp)` - Precision score
- `calculate_recall(tp, fn)` - Recall/sensitivity
- `calculate_f1(precision, recall)` - F1 score

## Structure
```
Lab1/
├── src/ml_metrics.py
├── test/test_pytest.py
├── test/test_unittest.py
├── workflows/pytest.yml
├── workflows/unittest.yml
└── requirements.txt
```

## Local Testing
```bash
# Pytest
pytest test/test_pytest.py -v

# Unittest
python -m unittest test.test_unittest -v
```

## CI/CD

GitHub Actions automatically runs both test suites on every push to main branch.
