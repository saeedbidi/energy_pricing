from pathlib import Path
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

class Plotter:
    @staticmethod
    def plot_predictions(y_test, predictions: dict, output_folder: str = 'output', title: str = 'Actual vs Predicted Energy Prices (Weekly Sample)') -> None:
        """
        Creates and saves a comparison plot between actual and predicted values, using weekly samples.

        Parameters:
        y_test (pd.Series): The actual values.
        predictions (dict): A dictionary of model names and their corresponding predicted values.
        output_folder (str): The folder where the plot will be saved (default is 'output').
        title (str): The title of the plot (optional).
        """
        # Get the root directory (main directory of the repo)
        root_dir = Path(__file__).resolve().parent.parent
        output_path = root_dir / output_folder  # Set output path relative to root directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Define different line styles to differentiate the models visually
        line_styles = ['--', ':', '-.']

        # Initialise the plot
        plt.figure(figsize=(10, 6))

        # Plot actual values
        plt.plot(y_test.index, y_test.values, label='Actual Prices', alpha=1, linewidth=2, linestyle='-')

        # Plot each model's predictions using different line styles
        for i, (model_name, preds) in enumerate(predictions.items()):
            plt.plot(y_test.index, preds, label=f'{model_name} Predicted Prices',
                     linewidth=2, linestyle=line_styles[i % len(line_styles)])

        # Customise the plot's appearance
        plt.legend()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Unit Price (inc VAT)')
        plt.xticks(rotation=45)
        plt.grid(True)

        # Save the plot to output folder
        plot_filename = output_path / 'predictions_plot.png'
        plt.savefig(plot_filename)
        plt.close()

        logging.info(f'Plot saved at {plot_filename}')
