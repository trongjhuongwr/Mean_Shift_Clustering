import os
import numpy as np
import argparse
from core.mean_shift import MeanShift
from core.preprocessing import load_data, normalize_data
from core.evaluation import calculate_silhouette_score, calculate_davies_bouldin_score
from core.visualization import plot_clusters

def main():
    parser = argparse.ArgumentParser(description='Mean-Shift Clustering')
    parser.add_argument('--bandwidth', type=float, default=0.5, help='Bán kính bandwidth')
    parser.add_argument('--kernel', type=str, default='flat', choices=['flat', 'gaussian'], help='Loại kernel')
    parser.add_argument('--data_path', type=str, default='data/raw/Customers.csv', help='Đường dẫn đến tệp dữ liệu')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Tệp dữ liệu không tồn tại: {args.data_path}")
        return

    os.makedirs('results/cluster_csv', exist_ok=True)
    os.makedirs('results/evaluation_metrics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    data = load_data(args.data_path)
    data = normalize_data(data)

    model = MeanShift(bandwidth=args.bandwidth, kernel=args.kernel)
    model.fit(data)

    labels = model.labels_
    centers = model.cluster_centers_

    sil_score = calculate_silhouette_score(data, labels)
    db_score = calculate_davies_bouldin_score(data, labels)

    np.savetxt('results/cluster_csv/clusters.csv', labels, delimiter=',')
    with open('results/evaluation_metrics/metrics.csv', 'w') as f:
        f.write(f'Silhouette Score,{sil_score if sil_score is not None else "N/A"}\n')
        f.write(f'Davies-Bouldin Score,{db_score if db_score is not None else "N/A"}\n')

    plot_clusters(data, labels, centers)

if __name__ == '__main__':
    main()