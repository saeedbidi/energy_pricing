from pathlib import Path
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO)

class Report:
    @staticmethod
    def evaluate_model(y_test, y_pred: np.ndarray, model_name: str, output_folder: str = 'output') -> tuple:
        """
        Evaluates the performance of a machine learning model and saves the results to a text file.

        Parameters:
        y_test (pd.Series): The true target values.
        y_pred (np.ndarray): The predicted target values.
        model_name (str): The name of the model being evaluated.
        output_folder (str): The folder where the report will be saved.

        Returns:
        tuple: RMSE, MAE, and R-squared scores for the model.
        """
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Get the root directory (main directory of the repo)
        root_dir = Path(__file__).resolve().parent.parent
        output_path = root_dir / output_folder  # Set output path relative to root directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a report file and save the results
        report_filename = output_path / f'{model_name}_evaluation_report.txt'
        with report_filename.open('w') as f:
            f.write(f'Performance of {model_name}:\n')
            f.write(f'Root Mean Squared Error (RMSE): {rmse}\n')
            f.write(f'Mean Absolute Error (MAE): {mae}\n')
            f.write(f'R-squared: {r2}\n')

        logging.info(f'Report saved at {report_filename}')

        return rmse, mae, r2
