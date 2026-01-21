# scripts/modeling.py
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib

def run_training_comparison(data_dir, model_dir):
    # Load Data
    X_train = np.load(f"{data_dir}/X_train_res.npy")
    y_train = np.load(f"{data_dir}/y_train_res.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    # CatBoost
    print("\n--- Melatih CatBoost ---")
    cb = CatBoostClassifier(iterations=1000, task_type="GPU", devices='0', verbose=100)
    cb.fit(X_train, y_train, eval_set=(X_test, y_test))
    cb.save_model(f"{model_dir}/catboost_model.cbm")

    # XGBoost 
    print("\n--- Melatih XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        tree_method='hist', # Algoritma histogram
        device='cuda',      # Mengaktifkan RTX 5060 Ti
        early_stopping_rounds=100,
        random_state=42
    )
    
    # Fit dengan tracking loss
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
    joblib.dump(xgb, f"{model_dir}/xgboost_model.pkl")

    # Simpan history XGBoost untuk visualisasi
    xgb_history = xgb.evals_result()
    joblib.dump(xgb_history, f"{model_dir}/xgb_history.pkl")

if __name__ == "__main__":
    run_training_comparison('../data/processed', '../models')