import numpy as np
from loguru import logger
from mlflow.pyfunc import PythonModel

class LatentFactorModel(PythonModel):
    def __init__(self, k, var_name='power', var_P='farmId', var_U='periodId', warm_start=False, verbose=False, learning_rate=None, random_state=42):
        """
        Initialize the LatentFactorModel object.

        Parameters:
        - k (int): The number of latent factors.
        - var_name (str): The name of the target variable.
        - var_P (str): The name of the column representing P.
        - var_U (str): The name of the column representing U.
        - warm_start (bool): Whether to perform warm start initialization.
        - verbose (bool): Whether to print verbose output during training.
        """
        self.k = k 
        self.var_name = var_name
        self.var_P = var_P
        self.var_U = var_U
        self.warm_start = warm_start
        self.verbose = verbose
        self.learning_rate=learning_rate
        self.random_state = random_state

    def initialize_matrix(self, n, k, warm_start):
        """
        Initialize a matrix with random values or warm start.

        Parameters:
        - n: int, number of rows in the matrix.
        - k: int, number of columns in the matrix.
        - warm_start: bool, whether to warm start the matrix.
        - values: array-like, initial values for the matrix.

        Returns:
        - matrix: numpy array, initialized matrix.
        """
        np.random.seed(self.random_state)
        matrix = np.random.rand(n, k)
        if warm_start:
            avg_values = np.mean(matrix, axis=1)
            matrix[:, 0] = avg_values
        return matrix

    def compute_rmse(self, targets, predictions):
        """
        Compute the Root Mean Squared Error (RMSE) between targets and predictions.

        Parameters:
        - targets: array-like, actual target values.
        - predictions: array-like, predicted values.

        Returns:
        - rmse: float, RMSE between targets and predictions.
        """
        try:
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            return rmse
        except Exception as e:
            logger.info(f"Error computing RMSE: {e}")

    def least_squares(self, list_ids, X, lambda_reg):
        """
        Perform least squares optimization.

        Parameters:
        - list_ids: array-like, list of indices and target values.
        - X: numpy array, matrix for the optimization.
        - lambda_reg: float, regularization parameter.

        Returns:
        - coefficients: numpy array, optimized coefficients.
        """
        try:
            indices = list_ids[:, 0].astype(int)
            X_subset = X[indices]
            XtX = np.dot(X_subset.T, X_subset) + lambda_reg * np.eye(X_subset.shape[1])
            XtY = np.dot(X_subset.T, list_ids[:, 1])
            return np.linalg.solve(XtX, XtY).T
        except Exception as e:
            logger.info(f"Error in least squares: {e}")
            return None

    def alternating_least_squares(self, P, n_epochs, train_df, valid_df, lambda_reg_U, lambda_reg_P):
        """
        Perform Alternating Least Squares optimization.

        Parameters:
        - P: numpy array, matrix P.
        - n_epochs: int, number of training epochs.
        - train_df: pandas DataFrame, training data.
        - valid_df: pandas DataFrame, validation data.
        - lambda_reg_U: float, regularization parameter for U.
        - lambda_reg_P: float, regularization parameter for P.

        Returns:
        - U, P, train_errors, valid_errors, train_preds, valid_preds: Tuple of results.
        """
        train_errors, valid_errors = [], []
        try:
            for epoch in range(n_epochs):
                U = np.vstack(train_df.groupby(self.var_U).
                                        apply(lambda x: self.least_squares(x[[self.var_P, self.var_name]].values, P, lambda_reg_P)))
                P = np.vstack(train_df.groupby(self.var_P).
                                        apply(lambda x: self.least_squares(x[[self.var_U, self.var_name]].values, U, lambda_reg_U)))
                train_preds = self.predict(U, P, train_df)
                train_error = self.compute_rmse(train_df[self.var_name].values, train_preds)
                train_errors.append(train_error)
                valid_preds = self.predict(U, P, valid_df)
                valid_error = self.compute_rmse(valid_df[self.var_name].values, valid_preds)
                valid_errors.append(valid_error)
                if self.verbose:
                    logger.info('____________________')
                    logger.info(f' Iteration: {epoch}')
                    logger.info(f' mse training error: {train_error}')
                    logger.info(f' mse validation error {valid_error}')
            return U, P
        except Exception as e:
            logger.info(f"Error in alternating least squares: {e}")
            return None

    def stochastic_gradient_descent(self, P, U, n_epochs, train_df, valid_df, lambda_reg_U, lambda_reg_P):
        """
        Perform Stochastic Gradient Descent optimization.

        Parameters:
        - P: numpy array, matrix P.
        - U: numpy array, matrix U.
        - n_epochs: int, number of training epochs.
        - train_df: pandas DataFrame, training data.
        - valid_df: pandas DataFrame, validation data.
        - learning_rate: float, learning rate.
        - lambda_reg_U: float, regularization parameter for U.
        - lambda_reg_P: float, regularization parameter for P.

        Returns:
        - U, P, train_errors, valid_errors, train_preds, valid_preds: Tuple of results.
        """
        train_errors, valid_errors = [], []
        try:
            for epoch in range(n_epochs):
                for row in train_df.itertuples():
                    u, p, target = row.__getattribute__(self.var_U), row.__getattribute__(self.var_P), row.__getattribute__(self.var_name)
                    prediction = np.dot(U[u], P[p])
                    error = target - prediction
                    U[u] += self.learning_rate * (error * P[p] - lambda_reg_U * U[u])
                    P[p] += self.learning_rate * (error * U[u] - lambda_reg_P * P[p])
                train_preds = self.predict(U, P, train_df)
                train_error = self.compute_rmse(train_df[self.var_name].values, train_preds)
                train_errors.append(train_error)
                valid_preds = self.predict(U, P, valid_df)
                valid_error = self.compute_rmse(valid_df[self.var_name].values, valid_preds)
                valid_errors.append(valid_error)
                if self.verbose:
                    logger.info('____________________')
                    logger.info(f' Iteration: {epoch}')
                    logger.info(f' mse training error: {train_error}')
                    logger.info(f' mse validation error {valid_error}')
            return U, P
        except Exception as e:
            logger.info(f"Error in stochastic gradient descent: {e}")
            return None

    def factorization(self, train_df, valid_df, solver='als', n_epochs=5, lambda_reg_P=50, lambda_reg_U=50):
        """
        Perform matrix factorization using ALS or SGD.

        Parameters:
        - train_df: pandas DataFrame, training data.
        - valid_df: pandas DataFrame, validation data.
        - solver: str, solver method ('als' or 'sgd').
        - n_epochs: int, number of training epochs.
        - lambda_reg_P: float, regularization parameter for P.
        - lambda_reg_U: float, regularization parameter for U.
        - learning_rate: float, learning rate for SGD.

        Returns:
        - U, P, train_errors, valid_errors, train_preds, valid_preds: Tuple of results.
        """
        try:
            n_p = len(train_df[self.var_P].unique())
            n_u = len(train_df[self.var_U].unique())
            P = self.initialize_matrix(n_p, self.k, self.warm_start)
            U = self.initialize_matrix(n_u, self.k, self.warm_start)
            if solver == 'als':
                U, P = self.alternating_least_squares(P=P, n_epochs = n_epochs, 
                                                        train_df=train_df, valid_df=valid_df, 
                                                        lambda_reg_U =lambda_reg_U, lambda_reg_P=lambda_reg_P)
            elif solver == 'sgd':



                U, P = self.stochastic_gradient_descent(P=P, U=U, n_epochs=n_epochs, 
                                                        train_df=train_df, valid_df=valid_df, 
                                                        lambda_reg_U=lambda_reg_U, lambda_reg_P=lambda_reg_P)
            return U, P
        except Exception as e:
            logger.info(f"Error in factorization: {e}")
            return None

    def predict(self, U, P, df):
        """
        Make predictions using matrices U and P.

        Parameters:
        - U: numpy array, matrix U.
        - P: numpy array, matrix P.
        - df: pandas DataFrame, input data.

        Returns:
        - predictions: array-like, predictions.
        """
        try:
            predictions = np.zeros(len(df))
            for i, row in enumerate(df.itertuples()):
                u, p = row.__getattribute__(self.var_U), row.__getattribute__(self.var_P)
                predictions[i] = np.dot(U[u], P[p])
            return predictions
        except Exception as e:
            logger.info(f"Error in prediction: {e}")
            return None

