import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


def predict_and_evaluate(X_test_path, model_path, output_path, true_labels_path):
    # Load preprocessed test data
    X_test = pd.read_csv("test_data_X.csv")

    X_test_cleaned = X_test[X_test["year"] <= 2020]
    X_test_cleaned["Car_Age"] = 2020 - X_test_cleaned["year"]
    X_test_cleaned = X_test_cleaned.dropna(axis=1)
    X_test_cleaned = X_test_cleaned.drop(columns=["year"])
    X_test_cleaned = pd.get_dummies(X_test_cleaned)
    # List of column names to check
    columns_to_check = [
        'mileage',	'tax',	'mpg',	'engineSize',	'Car_Age',
        'model_ 1 Series', 'model_ 2 Series', 'model_ 3 Series', 'model_ 4 Series', 
        'model_ 5 Series', 'model_ 6 Series', 'model_ 7 Series', 'model_ 8 Series', 
        'model_ A Class', 'model_ A1', 'model_ A2', 'model_ A3', 'model_ A4', 
        'model_ A5', 'model_ A6', 'model_ A7', 'model_ A8', 'model_ Accent', 
        'model_ Adam', 'model_ Agila', 'model_ Amarok', 'model_ Amica', 'model_ Ampera', 
        'model_ Antara', 'model_ Arteon', 'model_ Astra', 'model_ Auris', 'model_ Avensis', 
        'model_ Aygo', 'model_ B Class', 'model_ B-MAX', 'model_ Beetle', 'model_ C Class', 
        'model_ C-HR', 'model_ C-MAX', 'model_ CC', 'model_ CL Class', 'model_ CLA Class', 
        'model_ CLC Class', 'model_ CLK', 'model_ CLS Class', 'model_ Caddy', 
        'model_ Caddy Life', 'model_ Caddy Maxi', 'model_ Caddy Maxi Life', 
        'model_ California', 'model_ Camry', 'model_ Caravelle', 'model_ Cascada', 
        'model_ Citigo', 'model_ Combo Life', 'model_ Corolla', 'model_ Corsa', 
        'model_ Crossland X', 'model_ E Class', 'model_ EcoSport', 'model_ Edge', 
        'model_ Eos', 'model_ Escort', 'model_ Fabia', 'model_ Fiesta', 'model_ Focus', 
        'model_ Fox', 'model_ Fusion', 'model_ G Class', 'model_ GL Class', 
        'model_ GLA Class', 'model_ GLB Class', 'model_ GLC Class', 'model_ GLE Class', 
        'model_ GLS Class', 'model_ GT86', 'model_ GTC', 'model_ Galaxy', 'model_ Getz', 
        'model_ Golf', 'model_ Golf SV', 'model_ Grand C-MAX', 'model_ Grand Tourneo Connect', 
        'model_ Grandland X', 'model_ Hilux', 'model_ I10', 'model_ I20', 'model_ I30', 
        'model_ I40', 'model_ I800', 'model_ IQ', 'model_ IX20', 'model_ IX35', 
        'model_ Insignia', 'model_ Ioniq', 'model_ Jetta', 'model_ KA', 'model_ Ka+', 
        'model_ Kadjar', 'model_ Kamiq', 'model_ Karoq', 'model_ Kodiaq', 'model_ Kona', 
        'model_ Kuga', 'model_ Land Cruiser', 'model_ M Class', 'model_ M2', 'model_ M3', 
        'model_ M4', 'model_ M5', 'model_ M6', 'model_ Meriva', 'model_ Mokka', 
        'model_ Mokka X', 'model_ Mondeo', 'model_ Mustang', 'model_ Octavia', 
        'model_ PROACE VERSO', 'model_ Passat', 'model_ Polo', 'model_ Prius', 'model_ Puma', 
        'model_ Q2', 'model_ Q3', 'model_ Q5', 'model_ Q7', 'model_ Q8', 'model_ R Class', 
        'model_ R8', 'model_ RAV4', 'model_ RS3', 'model_ RS4', 'model_ RS5', 'model_ RS6', 
        'model_ RS7', 'model_ Ranger', 'model_ Rapid', 'model_ Roomster', 'model_ S Class', 
        'model_ S-MAX', 'model_ S3', 'model_ S4', 'model_ S5', 'model_ S8', 'model_ SL CLASS', 
        'model_ SLK', 'model_ SQ5', 'model_ SQ7', 'model_ Santa Fe', 'model_ Scala', 
        'model_ Scirocco', 'model_ Sharan', 'model_ Shuttle', 'model_ Streetka', 
        'model_ Superb', 'model_ Supra', 'model_ T-Cross', 'model_ T-Roc', 'model_ TT', 
        'model_ Terracan', 'model_ Tigra', 'model_ Tiguan', 'model_ Tiguan Allspace', 
        'model_ Touareg', 'model_ Touran', 'model_ Tourneo Connect', 'model_ Tourneo Custom', 
        'model_ Transit Tourneo', 'model_ Tucson', 'model_ Up', 'model_ Urban Cruiser', 
        'model_ V Class', 'model_ Vectra', 'model_ Veloster', 'model_ Verso', 'model_ Verso-S', 
        'model_ Viva', 'model_ Vivaro', 'model_ X-CLASS', 'model_ X1', 'model_ X2', 
        'model_ X3', 'model_ X4', 'model_ X5', 'model_ X6', 'model_ X7', 'model_ Yaris', 
        'model_ Yeti', 'model_ Yeti Outdoor', 'model_ Z3', 'model_ Z4', 'model_ Zafira', 
        'model_ Zafira Tourer', 'model_ i3', 'model_ i8', 'model_180', 'model_200', 
        'model_220', 'model_230', 'transmission_Automatic', 'transmission_Manual', 
        'transmission_Other', 'transmission_Semi-Auto', 'fuelType_Diesel', 
        'fuelType_Electric', 'fuelType_Hybrid', 'fuelType_Other', 'fuelType_Petrol', 
        'brand_audi', 'brand_bmw', 'brand_ford', 'brand_hyundai', 'brand_merc', 
        'brand_skoda', 'brand_toyota', 'brand_vauxhall', 'brand_vw'
    ]

    # Add missing columns with default value False
    for column in columns_to_check:
        if column not in X_test_cleaned.columns:
            X_test_cleaned[column] = False

    # Now df has all the required columns

    # Reorder the columns in the DataFrame according to columns_to_check
    X_test_cleaned = X_test_cleaned.reindex(columns=columns_to_check)

    # Now df has columns in the specified order
    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions
    y_pred_log = model.predict(X_test_cleaned)
    y_pred = np.exp(y_pred_log)

    # Save predictions
    predictions = pd.DataFrame(y_pred, columns=["price"])
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Load true labels
    y_true = pd.read_csv(true_labels_path)

    # Evaluate performance
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f'R² on test data: {r2}')
    print(f'Mean Squared Error on test data: {mse}')
    print(f'Mean Absolute Percentage Error on test data: {mape}')

if __name__ == "__main__":
    X_test_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    true_labels_path = sys.argv[4]
    predict_and_evaluate(X_test_path, model_path, output_path, true_labels_path)