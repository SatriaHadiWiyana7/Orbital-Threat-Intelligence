import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clean_column_names(df):
    """Menyeragamkan nama kolom menjadi huruf kecil untuk menghindari KeyError"""
    df.columns = [col.lower() for col in df.columns]
    return df

def plot_missing_values(df):
    """Visualisasi persentase data yang hilang"""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if not missing_pct.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='Reds_r')
        plt.title('Persentase Missing Values per Kolom')
        plt.xlabel('Persentase (%)')
        plt.show()
    return missing_pct