import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def load_data(file_path):
    """Tải dữ liệu từ tệp CSV và xử lý cơ bản.
    
    Args:
        file_path (str): Đường dẫn đến file CSV
        
    Returns:
        pd.DataFrame: DataFrame đã xử lý ban đầu
    """
    # Cố gắng đọc file với nhiều encoding khác nhau
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    
    # Xử lý giá trị null/nan
    df = df.fillna(df.mean(numeric_only=True))
    
    # Mã hóa các cột dạng chuỗi (categorical)
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    
    return df

def normalize_data(data, feature_range=(0, 1)):
    """Chuẩn hóa dữ liệu về khoảng feature_range.
    
    Args:
        data (pd.DataFrame hoặc np.ndarray): Dữ liệu cần chuẩn hóa
        feature_range (tuple): Khoảng giá trị đích (mặc định: (0, 1))
        
    Returns:
        np.ndarray: Dữ liệu đã chuẩn hóa
        MinMaxScaler: Đối tượng scaler để có thể inverse transform sau này
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    if isinstance(data, pd.DataFrame):
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def standardize_data(data):
    """Chuẩn tắc hóa dữ liệu (z-score).
    
    Args:
        data (pd.DataFrame hoặc np.ndarray): Dữ liệu cần chuẩn tắc hóa
        
    Returns:
        np.ndarray: Dữ liệu đã chuẩn tắc hóa
        StandardScaler: Đối tượng scaler để có thể inverse transform sau này
    """
    scaler = StandardScaler()
    if isinstance(data, pd.DataFrame):
        standardized_data = scaler.fit_transform(data)
    else:
        standardized_data = scaler.fit_transform(data)
    return standardized_data, scaler

def preprocess_data(file_path, method='normalize', feature_range=(0, 1)):
    """Hàm tổng hợp để tiền xử lý dữ liệu.
    
    Args:
        file_path (str): Đường dẫn đến file dữ liệu
        method (str): Phương pháp xử lý ('normalize' hoặc 'standardize')
        feature_range (tuple): Khoảng giá trị cho normalize (chỉ dùng với method='normalize')
    
    Returns:
        np.ndarray: Dữ liệu đã xử lý
        scaler: Đối tượng scaler tương ứng
        pd.DataFrame: DataFrame gốc
    """
    df = load_data(file_path)
    
    # Lưu dữ liệu gốc để tham chiếu sau này
    original_data = df.copy()
    
    if method == 'normalize':
        processed_data, scaler = normalize_data(df, feature_range)
    elif method == 'standardize':
        processed_data, scaler = standardize_data(df)
    else:
        raise ValueError("Phương pháp không được hỗ trợ. Sử dụng 'normalize' hoặc 'standardize'")
    
    return processed_data, scaler, original_data