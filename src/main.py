import os
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from core.mean_shift import MeanShift
from core.preprocessing import preprocess_data
from core.evaluation import calculate_silhouette_score, calculate_davies_bouldin_score
from core.visualization import plot_clusters

def estimate_bandwidth(data, quantile=0.3, n_samples=None):
    """Ước lượng bandwidth tự động dựa trên dữ liệu.
    
    Args:
        data (np.ndarray): Dữ liệu đầu vào
        quantile (float): Phân vị để ước lượng bandwidth
        n_samples (int): Số lượng mẫu để ước lượng (None = toàn bộ)
        
    Returns:
        float: Giá trị bandwidth ước lượng
    """
    from sklearn.neighbors import NearestNeighbors
    
    if n_samples is not None:
        idx = np.random.permutation(len(data))[:n_samples]
        data = data[idx]
    
    nbrs = NearestNeighbors(n_neighbors=min(len(data), 5)).fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # Tính khoảng cách trung bình đến k láng giềng gần nhất
    distances = np.sort(distances, axis=0)
    bandwidth = np.mean(distances[:, 1:])
    
    return bandwidth * quantile

def main():
    parser = argparse.ArgumentParser(description='Mean-Shift Clustering')
    parser.add_argument('--bandwidth', type=float, default=None, 
                        help='Bán kính bandwidth (None = tự động ước lượng)')
    parser.add_argument('--kernel', type=str, default='gaussian', 
                        choices=['flat', 'gaussian'], help='Loại kernel')
    parser.add_argument('--data_path', type=str, default='data/raw/Customers.csv', 
                        help='Đường dẫn đến tệp dữ liệu')
    parser.add_argument('--preprocess', type=str, default='normalize',
                        choices=['normalize', 'standardize'], 
                        help='Phương pháp tiền xử lý dữ liệu')
    parser.add_argument('--feature_range', type=str, default='0,1',
                        help='Khoảng giá trị cho normalize (min,max)')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Tệp dữ liệu không tồn tại: {args.data_path}")
        return

    # Tạo thư mục kết quả
    os.makedirs('results/cluster_csv', exist_ok=True)
    os.makedirs('results/evaluation_metrics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Phân tích feature_range
    feature_min, feature_max = map(float, args.feature_range.split(','))
    feature_range = (feature_min, feature_max)
    
    # Tiền xử lý dữ liệu
    data, scaler, original_df = preprocess_data(
        args.data_path, 
        method=args.preprocess, 
        feature_range=feature_range
    )
    
    # Ước lượng bandwidth nếu không được cung cấp
    if args.bandwidth is None:
        bandwidth = estimate_bandwidth(data)
        print(f"Bandwidth được ước lượng tự động: {bandwidth:.4f}")
    else:
        bandwidth = args.bandwidth
    
    # Huấn luyện mô hình
    model = MeanShift(bandwidth=bandwidth, kernel=args.kernel)
    model.fit(data)

    labels = model.labels_
    centers = model.cluster_centers_
    
    # Đánh giá mô hình
    sil_score = calculate_silhouette_score(data, labels)
    db_score = calculate_davies_bouldin_score(data, labels)
    
    # Thêm nhãn cụm vào dữ liệu gốc
    original_df['cluster'] = labels
    
    # Thống kê các cụm
    cluster_stats = original_df.groupby('cluster').agg(['mean', 'count'])
    
    # Lưu kết quả
    results_dir = os.path.join('results', os.path.basename(args.data_path).split('.')[0])
    os.makedirs(results_dir, exist_ok=True)
    
    original_df.to_csv(f'{results_dir}/clusters.csv', index=False)
    cluster_stats.to_csv(f'{results_dir}/cluster_statistics.csv')
    
    # Lưu metric đánh giá
    with open(f'{results_dir}/metrics.txt', 'w', encoding='utf-8') as f:
        f.write(f'Phương pháp tiền xử lý: {args.preprocess}\n')
        f.write(f'Bandwidth: {bandwidth}\n')
        f.write(f'Kernel: {args.kernel}\n')
        f.write(f'Số lượng cụm: {len(np.unique(labels))}\n')
        f.write(f'Silhouette Score: {sil_score:.4f}\n')
        f.write(f'Davies-Bouldin Score: {db_score:.4f}\n')
    
    # Trực quan hóa kết quả
    # Nếu có quá nhiều chiều, chọn 2 chiều đầu tiên để vẽ
    if data.shape[1] >= 2:
        # Sử dụng dữ liệu gốc sau khi đã tiền xử lý cho việc vẽ biểu đồ
        plot_clusters(data, labels, centers)
        
        # Vẽ thêm biểu đồ khác với các thuộc tính quan trọng (ví dụ: income vs age)
        if 'income' in original_df.columns and 'age' in original_df.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(original_df['income'], original_df['age'], c=labels, cmap='viridis', alpha=0.7)
            plt.title('Phân cụm khách hàng (Thu nhập vs Tuổi)')
            plt.xlabel('Thu nhập')
            plt.ylabel('Tuổi')
            plt.colorbar(scatter, label='Cụm')
            plt.savefig(f'{results_dir}/income_age_clusters.png')
            plt.close()
    
    print(f"Đã hoàn thành phân cụm với {len(np.unique(labels))} cụm")
    print(f"Kết quả đã được lưu vào thư mục: {results_dir}")

if __name__ == '__main__':
    main()