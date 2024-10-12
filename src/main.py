import pandas as pd
from sklearn.model_selection import train_test_split
from ml_model import MLModel
from report import Report
from plotter import Plotter
from data_preprocessing import preprocess_data

# Step 1: Data Loading and Preprocessing
data = preprocess_data('../data/csv_agile_C_London.csv')

# Define features (X) and target variable (y)
X = data[['hour', 'day_of_week', 'unit_price_lag_1']]
y = data['Unit Price (inc VAT)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Machine Learning Models
ml_model = MLModel()

# Train RandomForest and XGBoost models
ml_model.train('RandomForest', X_train, y_train)
ml_model.train('XGBoost', X_train, y_train)

# Make predictions with both models
rf_predictions = ml_model.predict('RandomForest', X_test)
xgb_predictions = ml_model.predict('XGBoost', X_test)

# Store predictions for later use (plotting, reporting)
predictions = {'RandomForest': rf_predictions, 'XGBoost': xgb_predictions}

# Step 3: Model Evaluation
report = Report()
report.evaluate_model(y_test, rf_predictions, 'RandomForest', output_folder='output')
report.evaluate_model(y_test, xgb_predictions, 'XGBoost', output_folder='output')

# Step 4: Plotting Results
plotter = Plotter()

# Resample the data to weekly frequency for a simpler plot
y_test_weekly = y_test.resample('W').first()
X_test_weekly = X_test.resample('W').first()

# Drop any rows with NaN after resampling
y_test_weekly.dropna(inplace=True)
X_test_weekly.dropna(inplace=True)

# Predict on the weekly sample
rf_predictions_weekly = ml_model.predict('RandomForest', X_test_weekly)
xgb_predictions_weekly = ml_model.predict('XGBoost', X_test_weekly)

# Plotting and saving results
plotter.plot_predictions(y_test_weekly, {'RandomForest': rf_predictions_weekly, 'XGBoost': xgb_predictions_weekly}, output_folder='output')

# Save evaluation report
report.evaluate_model(y_test, rf_predictions, 'RandomForest', output_folder='output')
report.evaluate_model(y_test, xgb_predictions, 'XGBoost', output_folder='output')
