from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import requests
from src.monitor import run_drift_report
from src.train import training

# DEFINE DAG SETTINGS
default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "start_date": datetime(2026, 2, 20),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds = 30),
}

# CREATE THE DAG
# This tells Airflow to run this pipeline once a day at midnight.
with DAG(
    "daily_churn_retraining_pipeline",
    default_args=default_args,
    description="Runs the PSI drift monitor on daily customer data, and retrains pipeline",
    schedule_interval="@daily", 
    catchup=False,
) as dag:

    # DEFINE THE TASKS
    # Task 1: A simple print statement to show the pipeline is starting
    def start_pipeline():
        print("Starting the Daily Drift Monitor Pipeline...")

    task_start = PythonOperator(
        task_id="start_pipeline",
        python_callable=start_pipeline,
    )

    # Task 2: The actual Machine Learning Monitor
    def run_monitor(**kwargs):
        # Replace this path with wherever your data_gen.py saves the daily files
        # data_path = "/opt/airflow/data/batches/batch_1_production.csv"
        drift_score_bool = run_drift_report()
        
        # Push the score to XCom (Airflow's memory bank) so the next task can read it
        kwargs['ti'].xcom_push(key='drift_score_bool', value = drift_score_bool)

    task_monitor = PythonOperator(
        task_id="run_psi_drift_check",
        python_callable=run_monitor,
    )

    def decide_whether_to_retrain(**kwargs):
        # Pull the drift score from the previous task's XCom
        drift_score_bool = kwargs['ti'].xcom_pull(task_ids='run_psi_drift_check', key='drift_score_bool')
        
        # The threshold decision
        if drift_score_bool:
            print("Drift is HIGH. Triggering retraining.")
            return "retrain_model" # Returns the exact task_id to run next
        else:
            print("Drift is LOW. Skipping retraining.")
            return "skip_retraining" # Returns the exact task_id to run next

    task_branch = BranchPythonOperator(
        task_id="check_drift_threshold",
        python_callable=decide_whether_to_retrain,
    )

    # TASK 3a: The Retraining Path
    def trigger_training():
        training("Churn") 
        
        print("Training complete. New model saved to /models directory.")

    task_retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=trigger_training,
    )

    def reload_fastapi():
        print("Notifying FastAPI to drop the old model and load the new one...")
        
        # Notice we use 'fastapi' instead of 'localhost'!
        url = "http://fastapi:8000/reload-model"
        
        try:
            response = requests.post(url)
            response.raise_for_status() # Raises an error if the request fails
            print(f"FastAPI Response: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Failed to contact FastAPI: {e}")
            raise # Fail the Airflow task so we get an alert

    task_reload_api = PythonOperator(
        task_id="reload_fastapi",
        python_callable=reload_fastapi,
    )

    # TASK 3b: The Skip Path (EmptyOperator just acts as a placeholder)
    task_skip = EmptyOperator(
        task_id="skip_retraining"
    )

    # Set the order of operations
    task_start >> task_monitor >> task_branch
    task_branch >> task_retrain >> task_reload_api
    task_branch >> task_skip