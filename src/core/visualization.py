import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

def plot_clusters(data, labels, centers=None, save_path=None):
    """Vẽ biểu đồ phân cụm cho hai đặc trưng đầu tiên.
    
    Args:
        data (np.ndarray): Dữ liệu đã được tiền xử lý
        labels (np.ndarray): Nhãn cụm
        centers (np.ndarray): Tâm các cụm
        save_path (str): Đường dẫn để lưu hình (None = không lưu)
    """
    plt.figure(figsize=(10, 8))
    
    # Nếu dữ liệu có nhiều hơn 2 chiều, sử dụng PCA để giảm về 2 chiều
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        if centers is not None:
            centers_2d = pca.transform(centers)
        else:
            centers_2d = None
            
        plt.title(f'Trực quan hóa phân cụm với PCA (giải thích {pca.explained_variance_ratio_.sum()*100:.2f}% phương sai)')
    else:
        data_2d = data
        centers_2d = centers
        plt.title('Trực quan hóa phân cụm')
    
    # Vẽ các điểm dữ liệu với màu tương ứng với nhãn cụm
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', 
                         alpha=0.7, s=40, edgecolors='w', linewidth=0.5)
    
    # Vẽ tâm cụm
    if centers_2d is not None and len(centers_2d) > 0:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', 
                   s=200, alpha=1, edgecolors='k', linewidth=2, label='Tâm cụm')
    
    # Thêm color bar
    plt.colorbar(scatter, label='Cụm')
    plt.xlabel('Thành phần chính 1' if data.shape[1] > 2 else 'Đặc trưng 1')
    plt.ylabel('Thành phần chính 2' if data.shape[1] > 2 else 'Đặc trưng 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu hình ảnh tại: {save_path}")
    
    plt.show()

def plot_feature_importance(df, labels, save_path=None):
    """Vẽ biểu đồ tầm quan trọng của các đặc trưng đối với việc phân cụm.
    
    Args:
        df (pd.DataFrame): DataFrame gốc chứa dữ liệu
        labels (np.ndarray): Nhãn cụm
        save_path (str): Đường dẫn để lưu hình (None = không lưu)
    """
    # Thêm nhãn cụm vào DataFrame
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    # Tính F-statistic cho từng đặc trưng
    feature_importance = {}
    for feature in df.columns:
        if df[feature].dtype in [np.int64, np.float64]:
            # Tính tổng bình phương giữa các nhóm (between-group sum of squares)
            group_means = df_with_labels.groupby('cluster')[feature].mean()
            overall_mean = df_with_labels[feature].mean()
            between_ss = sum(len(df_with_labels[df_with_labels['cluster'] == i]) * 
                            (mean - overall_mean)**2 
                            for i, mean in group_means.items())
            
            # Tính tổng bình phương trong các nhóm (within-group sum of squares)
            within_ss = sum(sum((df_with_labels[df_with_labels['cluster'] == i][feature] - mean)**2) 
                          for i, mean in group_means.items())
            
            if within_ss > 0:
                f_stat = (between_ss / (len(group_means) - 1)) / (within_ss / (len(df_with_labels) - len(group_means)))
                feature_importance[feature] = f_stat
            else:
                feature_importance[feature] = 0
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sắp xếp theo tầm quan trọng
    sorted_idx = np.argsort(importance)
    plt.barh([features[i] for i in sorted_idx], [importance[i] for i in sorted_idx])
    
    plt.xlabel('F-statistic')
    plt.ylabel('Đặc trưng')
    plt.title('Tầm quan trọng của các đặc trưng trong việc phân biệt các cụm')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_cluster_profiles(df, labels, save_path=None):
    """Vẽ biểu đồ profile cho từng cụm.
    
    Args:
        df (pd.DataFrame): DataFrame gốc chứa dữ liệu
        labels (np.ndarray): Nhãn cụm
        save_path (str): Đường dẫn để lưu hình (None = không lưu)
    """
    # Thêm nhãn cụm vào DataFrame
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    # Tính giá trị trung bình của từng cụm cho mỗi đặc trưng số
    numeric_features = df.select_dtypes(include=[np.number]).columns
    cluster_profiles = df_with_labels.groupby('cluster')[numeric_features].mean()
    
    # Chuẩn hóa giá trị để dễ so sánh
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_profiles = pd.DataFrame(
        scaler.fit_transform(cluster_profiles),
        index=cluster_profiles.index,
        columns=cluster_profiles.columns
    )
    
    # Vẽ heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(normalized_profiles, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    plt.title('Đặc trưng của các cụm (đã chuẩn hóa)')
    plt.ylabel('Cụm')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Vẽ bar chart cho các cụm
    n_clusters = len(np.unique(labels))
    n_features = len(numeric_features)
    
    plt.figure(figsize=(15, n_clusters * 3))
    for i, cluster in enumerate(sorted(np.unique(labels))):
        plt.subplot(n_clusters, 1, i + 1)
        cluster_profile = normalized_profiles.loc[cluster]
        plt.barh(range(n_features), cluster_profile, color='skyblue')
        plt.yticks(range(n_features), numeric_features)
        plt.title(f'Cụm {cluster} (n={sum(labels == cluster)})')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        base_path = save_path.rsplit('.', 1)[0]
        plt.savefig(f"{base_path}_barchart.png", dpi=300, bbox_inches='tight')
    
    plt.show()