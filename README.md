[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14187972.svg)](https://doi.org/10.5281/zenodo.14187972)
![Status](https://img.shields.io/badge/status-development-orange)

# Collaborative-Data-Imputation

<img src="img/colab_data_imputation.png" alt="Image Alt Text" width="700"/>

### Reconstruction of wind power data

In power system operations and electricity markets, missing data is a common problem in practice. This issue is especially significant when large-scale data-driven methods are used for point or probabilistic wind power forecasting. Data imputation methods, such as k-nearest neighbors and factor models, are crucial for filling in missing values before training forecasting models. These techniques ensure data completeness, which is essential for the accuracy of data-driven forecasting approaches.

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
