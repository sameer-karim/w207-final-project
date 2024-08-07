import pandas as pd
import numpy as np
import joblib
import sys


def predict( model_path, output_path):
    # Load preprocessed data
    data = pd.read_csv( "preprocessed_data.csv")

    # Separate features and price column
    X_test = data.drop(columns=["price"])

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)

    # Save predictions
    predictions = pd.DataFrame(y_pred, columns=["price"])
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    predict(model_path, output_path)