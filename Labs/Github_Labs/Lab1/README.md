# Lab 1 - ML Metrics Calculator with MLOps Pipeline

**Author:** Santhoshkumar Chandrasekar  
**Course:** DADS7305 - MLOps  
**Semester:** Spring 2026  
**Modified Lab:** Lab 1 with ML/Data Science Focus

## Overview

This project demonstrates MLOps principles by implementing an **ML Metrics Calculator** instead of a basic arithmetic calculator. The calculator computes common machine learning evaluation metrics used for binary classification model assessment.

## Modifications from Original Lab

**Original Lab:**
- Basic arithmetic calculator with functions: add, subtract, multiply
- Simple calculator.py with basic math operations

**Modified Version:**
- ML metrics calculator for model evaluation
- Data science-focused implementation
- Real-world application for ML engineers

**Functions Implemented:**
1. `calculate_accuracy(tp, tn, fp, fn)` - Overall model correctness
2. `calculate_precision(tp, fp)` - Positive predictive value
3. `calculate_recall(tp, fn)` - Sensitivity/True positive rate
4. `calculate_f1_score(precision, recall)` - Harmonic mean of precision and recall

## Project Structure
```
Lab1/
├── src/
│   ├── ml_metrics.py       # Main ML metrics calculator
│   └── __init__.py
├── test/
│   ├── test_pytest.py      # Pytest test suite
│   ├── test_unittest.py    # Unittest test suite
│   └── __init__.py
├── workflows/
│   ├── ml_metrics_pytest.yml
│   └── ml_metrics_unittest.yml
├── data/                    # Placeholder for datasets
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git installed
- Virtual environment tool

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/santhoshkchandrasekar/MLOps.git
cd MLOps/Labs/Github_Labs/Lab1
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv lab_01
lab_01\Scripts\activate

# Mac/Linux
python -m venv lab_01
source lab_01/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running Tests Locally

### Using Pytest
```bash
pytest test/test_pytest.py -v
```

**Expected Output:**
```
test_accuracy_perfect_prediction PASSED
test_accuracy_half_correct PASSED
test_precision_perfect PASSED
test_precision_half PASSED
test_recall_perfect PASSED
test_recall_half PASSED
test_f1_score_perfect PASSED
test_f1_score_balanced PASSED
test_edge_case_zero_division PASSED
```

### Using Unittest
```bash
python -m unittest test.test_unittest -v
```

## CI/CD Pipeline with GitHub Actions

This project implements automated testing using GitHub Actions:

### Workflows:
1. **ML Metrics Pytest Workflow** (`ml_metrics_pytest.yml`)
   - Triggers on push/pull request to main branch
   - Runs all pytest tests
   - Generates XML test reports
   - Uploads test artifacts

2. **ML Metrics Unittest Workflow** (`ml_metrics_unittest.yml`)
   - Parallel testing with unittest framework
   - Validates all test cases
   - Provides success/failure notifications

### How It Works:
- Every push to `main` branch triggers both workflows
- Tests run in Ubuntu environment with Python 3.8
- Dependencies installed from `requirements.txt`
- Test results visible in GitHub Actions tab

## Example Usage
```python
from src.ml_metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score
)

# Example: Confusion matrix from a binary classifier
true_positives = 85    # Correctly predicted positive cases
true_negatives = 90    # Correctly predicted negative cases
false_positives = 10   # Incorrectly predicted as positive
false_negatives = 15   # Incorrectly predicted as negative

# Calculate metrics
accuracy = calculate_accuracy(true_positives, true_negatives, 
                              false_positives, false_negatives)
precision = calculate_precision(true_positives, false_positives)
recall = calculate_recall(true_positives, false_negatives)
f1 = calculate_f1_score(precision, recall)

print(f"Model Performance Metrics:")
print(f"Accuracy:  {accuracy:.2%}")   # 87.50%
print(f"Precision: {precision:.2%}")  # 89.47%
print(f"Recall:    {recall:.2%}")     # 85.00%
print(f"F1 Score:  {f1:.2%}")         # 87.18%
```

## Key MLOps Concepts Demonstrated

1. ✅ **Version Control:** Git and GitHub for collaborative development
2. ✅ **Environment Management:** Virtual environments for dependency isolation
3. ✅ **Testing Frameworks:** Both pytest and unittest implementations
4. ✅ **Continuous Integration:** Automated testing with GitHub Actions
5. ✅ **Code Quality:** Comprehensive test coverage with edge cases
6. ✅ **Documentation:** Clear README and inline code documentation

## What Makes This Different

Unlike basic calculator implementations, this project:
- Uses real-world ML evaluation metrics
- Handles edge cases (zero division)
- Provides practical value for data scientists
- Demonstrates domain knowledge in ML
- Shows understanding of model evaluation

## Learning Outcomes

Through this lab, I demonstrated:
- Setting up Python virtual environments
- Writing clean, documented Python code
- Implementing comprehensive unit tests
- Configuring CI/CD pipelines
- Using Git for version control
- Understanding ML model evaluation metrics

## Contact

**Santhosh K Chandrasekar**  
MS Data Science, Northeastern University  
Email: chandrasekar.sa@northeastern.edu  
GitHub: [@santhoshkchandrasekar](https://github.com/santhoshkchandrasekar)

---

**Note:** This is a modified version of Lab 1 for the DADS7305 MLOps course. The original lab focused on basic arithmetic operations, while this version applies MLOps principles to machine learning metrics calculation.