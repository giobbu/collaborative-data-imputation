# Collaborative-Data-Imputation

<img src="img/colab_data_imputation.png" alt="Image Alt Text" width="700"/>

### Running the MLflow Experiment Script with Poetry

1. **Install Poetry**

    ```bash
    pip install poetry
    ```

2. **Install Project Dependencies** 

    Install the project dependencies, including MLflow, by running the following command in your project directory:

    ```bash
    poetry install
    ```

3. **Running the Experiments** 

    To run the `experiment.py` script within the Poetry environment, use the following command:

    ```bash
    poetry run python experiment.py
    ```

4. **Viewing MLflow Tracking**: 

    ```bash
    mlflow ui
    ```

After starting the MLflow UI, open your browser and go to `http://127.0.0.1:5000` to view experiment results, including parameters, metrics, and artifacts.