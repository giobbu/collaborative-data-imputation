# Collaborative-Data-Imputation

To run the `experiment.py` script using MLflow, follow these steps:

1. **Install MLflow**: If you haven't already installed MLflow, you can do so via pip:

    ```bash
    pip install mlflow
    ```

2. **Set up your environment**: Make sure you have your environment set up with the necessary dependencies.

3. **Run the script with MLflow**: Use the following command to run `experiment.py` with MLflow:

    ```bash
    mlflow server --host 127.0.0.1 --port 8080
    mlflow run experiment.py
    ```

This command will execute `experiment.py` with MLflow tracking enabled, allowing you to log metrics, parameters, and artifacts for easy experiment tracking and management.
