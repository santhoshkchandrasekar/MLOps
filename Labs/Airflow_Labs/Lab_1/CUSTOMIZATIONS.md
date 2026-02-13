# Lab 1 Customizations by Santhosh Chandrasekar

## Summary
This lab implements a K-Means clustering pipeline using Apache Airflow with significant customizations from the original template.

## Key Customizations

### 1. **Dataset Change**
- **Original:** Credit card customer data (file.csv with BALANCE, PURCHASES, CREDIT_LIMIT)
- **Custom:** Iris flower dataset (iris.csv with sepal_length, sepal_width, petal_length, petal_width)
- **Impact:** Changed from 3 financial features to 4 botanical features

### 2. **DAG Configuration**
- **Owner:** Changed from 'your_name' to 'santhosh_chandrasekar'
- **DAG Name:** Changed from 'Airflow_Lab1' to 'Santhosh_MLOps_Lab1'
- **Description:** Custom description "Custom MLOps Lab 1 - K-Means Clustering Pipeline by Santhosh"
- **Start Date:** Updated to 2026-02-13 (current date)
- **Schedule:** Added '@daily' schedule interval
- **Retries:** Changed from 0 to 1
- **Retry Delay:** Changed from 5 minutes to 3 minutes

### 3. **Model Configuration**
- **K-Range:** Changed from range(1, 50) to range(1, 10) - more appropriate for Iris dataset
- **Model Filename:** Changed from 'model.sav' to 'santhosh_kmeans_model.pkl'

### 4. **Code Improvements**
- Fixed import issue: Changed from `airflow.providers.standard.operators.python` to `airflow.operators.python`
- Updated all data preprocessing to handle Iris dataset features
- Modified prediction logic to work with Iris dataset

## Results
- **Optimal Clusters Found:** 3 (correctly identifies the 3 Iris species)
- **Pipeline Status:** All 4 tasks completed successfully
- **Execution Time:** ~8 seconds

## Evidence
- DAG runs successfully with custom name visible in Airflow UI
- Logs confirm "Loading Iris dataset for clustering"
- Final output: "Optimal no. of clusters for Iris dataset: 3"