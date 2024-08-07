import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib
import sys

def train_model(data_path, model_output):
    data = pd.read_csv(data_path)

    X = data.drop(columns=["price"])
    y = np.log(data['price'])

    # Split the data
    X_train_val, X_test, y_train_val_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train_log, y_val_log = train_test_split(X_train_val, y_train_val_log, test_size=0.25, random_state=42)

    # Train model
    lr = LinearRegression().fit(X_train_val, y_train_val_log)

    # Save the trained model
    joblib.dump(lr, model_output)

    # Evaluate the model
    y_pred_test_log = lr.predict(X_test)
    y_pred_test = np.exp(y_pred_test_log)
    y_test = np.exp(y_test_log).reset_index(drop=True)

    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)

    print(f'R² on test data: {r2}')
    print(f'Mean Squared Error on test data: {mse}')
    print(f'Mean Absolute Percentage Error on test data: {mape}')

if __name__ == "__main__":
    data_path = sys.argv[1]
    train_model(data_path, "pretrained_model.joblib")
