import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import logging

logging.basicConfig(level=logging.INFO)


class MLModel:
    def __init__(self):
        # Initialize models
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        }
        self.trained_models = {}  # Store trained models
        self.predictions = {}  # Store predictions from models

    def train(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains a machine learning model.

        Parameters:
        model_name (str): The name of the model to train.
        X_train (np.ndarray): Feature data for training.
        y_train (np.ndarray): Target data for training.
        """
        if model_name not in self.models:
            raise ValueError(f'Model "{model_name}" is not available.')

        model = self.models[model_name]
        try:
            logging.info(f'Training {model_name}...')
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model  # Save trained model
            logging.info(f'{model_name} training finished.')
        except Exception as e:
            logging.error(f'Error occurred during {model_name} training: {str(e)}')

    def predict(self, model_name: str, X_test: np.ndarray) -> np.ndarray:
        """
        Uses a trained model to make predictions.

        Parameters:
        model_name (str): The name of the model to use for prediction.
        X_test (np.ndarray): Feature data to predict on.

        Returns:
        np.ndarray: Predicted values for the test data.
        """
        if model_name not in self.trained_models:
            raise ValueError(f'Model "{model_name}" has not been trained yet.')

        logging.info(f'Making predictions with {model_name}...')
        model = self.trained_models[model_name]
        predictions = model.predict(X_test)
        self.predictions[model_name] = predictions
        return predictions
