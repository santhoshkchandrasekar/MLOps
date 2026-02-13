# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
# The old airflow.configuration API is deprecated.

# Define default arguments for your DAG
default_args = {
    'owner': 'santhosh_chandrasekar',  # CUSTOMIZATION 1: Your name
    'start_date': datetime(2026, 2, 13),  # CUSTOMIZATION 2: Today's date
    'retries': 1,  # CUSTOMIZATION 3: Changed from 0 to 1 retry
    'retry_delay': timedelta(minutes=3),  # CUSTOMIZATION 4: Changed from 5 to 3 minutes
}

# Create a DAG instance named 'Santhosh_MLOps_Lab1' with the defined default arguments
with DAG(
    'Santhosh_MLOps_Lab1',  # CUSTOMIZATION 5: Changed DAG name
    default_args=default_args,
    description='Custom MLOps Lab 1 - K-Means Clustering Pipeline by Santhosh',  # CUSTOMIZATION 6: Custom description
    schedule_interval='@daily',  # CUSTOMIZATION 7: Added daily schedule
    catchup=False,
) as dag:

    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "santhosh_kmeans_model.pkl"],  # CUSTOMIZATION 8: Custom model filename
    )

    # Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["santhosh_kmeans_model.pkl", build_save_model_task.output],  # CUSTOMIZATION 9: Custom model filename
    )

    # Set task dependencies
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()