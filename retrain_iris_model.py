import pandas as pd
import os
from datetime import datetime
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

def retrain_and_archive_model(data_path):
    
    ## Archive current model
    original_name = 'final_iris_model.keras'
    date_str = datetime.now().strftime("%Y%m%d")
    archived_name = f'final_iris_model_{date_str}.keras'
    if os.path.exists(original_name):
        os.rename(original_name, archived_name)
        print(f"Archived old model as: {archived_name}")
    else:
        raise FileNotFoundError(f"Original model {original_name} not found")
    
    ## Load new dataset for retraining
    df = pd.read_csv(data_path)
    X_train, y_train = df.drop('target', axis=1).values, df['target'].values
    # data = load_iris()
    # X_train, y_train = data.data, data.target
    
    ## Load model, retrain, and save with original name
    scaler = joblib.load("iris_scaler.pkl")
    model = load_model(archived_name)
    
    X_train_scaled = scaler.transform(X_train)
    y_train_encoded = to_categorical(y_train, num_classes=3)
    
    model.fit(X_train_scaled, y_train_encoded, epochs=5)
    model.save(original_name)
    return

if __name__ == "__main__":
    retrain_and_archive_model('./data/iris.csv')