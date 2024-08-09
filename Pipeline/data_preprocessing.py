import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_dir):
    # Load individual CSVs and add a 'brand' column
    audi = pd.read_csv(os.path.join(data_dir, 'audi.csv'))
    audi['brand'] = 'audi'
    bmw = pd.read_csv(os.path.join(data_dir, 'bmw.csv'))
    bmw['brand'] = 'bmw'
    ford = pd.read_csv(os.path.join(data_dir, 'ford.csv'))
    ford['brand'] = 'ford'
    hyundai = pd.read_csv(os.path.join(data_dir, 'hyundi.csv'))
    hyundai['brand'] = 'hyundai'
    hyundai.rename(columns={'tax(£)': 'tax'}, inplace=True)
    merc = pd.read_csv(os.path.join(data_dir, 'merc.csv'))
    merc['brand'] = 'merc'
    skoda = pd.read_csv(os.path.join(data_dir, 'skoda.csv'))
    skoda['brand'] = 'skoda'
    toyota = pd.read_csv(os.path.join(data_dir, 'toyota.csv'))
    toyota['brand'] = 'toyota'
    vauxhall = pd.read_csv(os.path.join(data_dir, 'vauxhall.csv'))
    vauxhall['brand'] = 'vauxhall'
    vw = pd.read_csv(os.path.join(data_dir, 'vw.csv'))
    vw['brand'] = 'vw'

    # Combine datasets
    full_datasets = pd.concat([audi, bmw, ford, hyundai, merc, skoda, toyota, vauxhall, vw])

    # Data cleaning and preprocessing
    full_datasets_cleaned = full_datasets[full_datasets["year"] <= 2020]
    full_datasets_cleaned["Car_Age"] = 2020 - full_datasets_cleaned["year"]
    full_datasets_cleaned = full_datasets_cleaned.dropna(axis=1)
    full_datasets_cleaned = full_datasets_cleaned.drop(columns=["year"])
    full_datasets_cleaned = pd.get_dummies(full_datasets_cleaned)

    return (full_datasets, full_datasets_cleaned)

def split_and_save_datasets(data, test_path_X, test_path_Y):
    # Split the data into features (X) and target (y)
    X = data.drop(columns=["price"])
    y = data['price']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save to CSV
    X_test.to_csv(test_path_X, index=False)
    y_test.to_csv(test_path_Y, index=False)

if __name__ == "__main__":
    data_dir = "100,000 UK Used Car Data set"
    test_path_X = "test_data_X.csv"
    test_path_Y = "test_data_Y.csv"

    (full_datasets,full_datasets_cleaned) = load_and_preprocess_data(data_dir)
    full_datasets_cleaned.to_csv("preprocessed_full_data_ready_for_model.csv", index=False)
    split_and_save_datasets(full_datasets, test_path_X, test_path_Y)
    
    print(f"Test data without target price saved to {test_path_X}")
    print(f"Test data car price saved to {test_path_Y}")
