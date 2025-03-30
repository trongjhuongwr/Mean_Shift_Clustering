import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Tải dữ liệu từ tệp CSV."""
    df = pd.read_csv(file_path)
    # Mã hóa cột 'gender'
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    return df.values

def normalize_data(data):
    """Chuẩn hóa dữ liệu về khoảng [0, 1]."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

def standardize_data(data):
    """Chuẩn tắc hóa dữ liệu (z-score)."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std