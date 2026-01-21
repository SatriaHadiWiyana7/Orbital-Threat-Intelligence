import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def run_preprocessing(input_path, output_dir):
    print("--- Memulai Tahap Preprocessing ---")
    
    # Load Data dengan penanganan tipe campuran
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [col.lower() for col in df.columns] 
    
    # Drop fitur yang tidak relevan atau hampir kosong (>80% missing)
    cols_to_drop = [
        'prefix', 'name', 'albedo', 'diameter_sigma', 'diameter', 
        'id', 'spkid', 'full_name', 'pdes', 'orbit_id', 'equinox'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Handling Redundancy berdasarkan temuan Heatmap
    redundant_cols = ['ad', 'q', 'per_y', 'epoch_cal', 'epoch_mjd']
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])
    
    # Target Cleaning
    df = df.dropna(subset=['pha'])
    
    # Encoding Label Target
    df['pha'] = df['pha'].map({'Y': 1, 'N': 0})
    if 'neo' in df.columns:
        df['neo'] = df['neo'].map({'Y': 1, 'N': 0})
    
    # Imputasi Sisa Missing Values (Hanya pada kolom numerik)
    df = df.fillna(df.median(numeric_only=True))
    
    # Penanganan Kolom Kategorikal Tersisa (jika ada)
    # Jika masih ada kolom object selain target, kita hapus atau encode
    df = df.select_dtypes(exclude=['object'])
    
    # Split Features & Target
    X = df.drop('pha', axis=1)
    y = df['pha']
    
    # Train-Test Split (Stratified karena imbalance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE untuk Balancing Data (Akselerasi CPU)
    print("Menyeimbangkan data dengan SMOTE")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Simpan Output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    np.save(f"{output_dir}/X_train_res.npy", X_resampled)
    np.save(f"{output_dir}/y_train_res.npy", y_resampled)
    np.save(f"{output_dir}/X_test.npy", X_test_scaled)
    np.save(f"{output_dir}/y_test.npy", y_test.values)
    
    print(f"Preprocessing Selesai! Fitur terpilih: {list(X.columns)}")
    print(f"Jumlah baris setelah SMOTE: {len(X_resampled)}")

if __name__ == "__main__":
    run_preprocessing(
        input_path='../data/raw/dataset.csv', 
        output_dir='../data/processed'
    )