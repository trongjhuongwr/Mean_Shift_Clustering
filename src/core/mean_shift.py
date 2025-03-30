import numpy as np
from .utils import euclidean_distance, flat_kernel, gaussian_kernel
from sklearn.metrics.pairwise import euclidean_distances

class MeanShift:
    def __init__(self, bandwidth=None, kernel='gaussian', max_iter=300, tol=1e-3, 
                 bin_seeding=False, min_bin_freq=1, cluster_all=True):
        """Khởi tạo mô hình Mean-Shift.
        
        Args:
            bandwidth (float): Bán kính để tính trung bình (None = tự động ước lượng)
            kernel (str): Loại kernel ('flat' hoặc 'gaussian')
            max_iter (int): Số lần lặp tối đa
            tol (float): Ngưỡng hội tụ
            bin_seeding (bool): Có sử dụng bin seeding để tăng tốc hay không
            min_bin_freq (int): Tần suất tối thiểu của các bin khi sử dụng bin_seeding
            cluster_all (bool): Có gán tất cả điểm vào cụm hay không
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        
        # Chọn kernel function dựa trên loại kernel
        if kernel == 'flat':
            self.kernel_func = flat_kernel
        elif kernel == 'gaussian':
            self.kernel_func = gaussian_kernel
        else:
            raise ValueError("Kernel không được hỗ trợ")
    
    def _estimate_bandwidth(self, data, quantile=0.3, n_samples=1000):
        """Ước lượng bandwidth tự động từ dữ liệu."""
        from sklearn.neighbors import NearestNeighbors
        
        if n_samples is not None and len(data) > n_samples:
            idx = np.random.permutation(len(data))[:n_samples]
            data = data[idx]
            
        nbrs = NearestNeighbors(n_neighbors=min(len(data), 5)).fit(data)
        distances, _ = nbrs.kneighbors(data)
        
        distances = np.sort(distances, axis=0)
        bandwidth = np.mean(distances[:, 1:])
        
        return bandwidth * quantile
    
    def _seeding(self, data):
        """Chọn các điểm ban đầu thông minh để tối ưu hóa thuật toán."""
        if not self.bin_seeding:
            return data.copy()
            
        # Tìm min, max cho mỗi chiều
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Tạo lưới bin
        n_features = data.shape[1]
        bins = np.linspace(0, 1, int(1 / self.bandwidth), endpoint=False)
        bin_edges = bins * (max_vals - min_vals).reshape(1, n_features) + min_vals.reshape(1, n_features)
        
        # Gán mỗi điểm vào bin
        bin_indices = np.floor((data - min_vals) / (max_vals - min_vals) / self.bandwidth).astype(int)
        bin_indices = np.clip(bin_indices, 0, len(bins) - 1)
        
        # Tính bin representation
        bin_counts = {}
        for i, point in enumerate(data):
            bin_key = tuple(bin_indices[i])
            if bin_key in bin_counts:
                bin_counts[bin_key].append(i)
            else:
                bin_counts[bin_key] = [i]
        
        # Chọn các bin có frequency > min_bin_freq
        seeds = []
        for bin_key, indices in bin_counts.items():
            if len(indices) >= self.min_bin_freq:
                mean_point = np.mean(data[indices], axis=0)
                seeds.append(mean_point)
        
        if len(seeds) == 0:
            return data.copy()
        
        return np.array(seeds)
    
    def fit(self, data):
        """Huấn luyện mô hình Mean-Shift trên dữ liệu.
        
        Args:
            data (np.ndarray): Dữ liệu đầu vào (n_samples, n_features)
        
        Returns:
            self: Đối tượng đã được huấn luyện
        """
        # Kiểm tra dữ liệu đầu vào
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Ước lượng bandwidth nếu chưa cung cấp
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(data)
            print(f"Bandwidth được ước lượng tự động: {self.bandwidth:.4f}")
        
        # Chọn các điểm ban đầu
        if self.bin_seeding:
            seeds = self._seeding(data)
        else:
            seeds = data.copy()
        
        # Thực hiện thuật toán mean-shift
        centroids = seeds.copy()
        for it in range(self.max_iter):
            print(f"Lặp {it+1}/{self.max_iter}, đang xử lý {len(centroids)} centroids", end='\r')
            new_centroids = []
            
            for i, centroid in enumerate(centroids):
                weights = []
                weighted_sum = np.zeros(data.shape[1])
                weight_sum = 0
                
                # Tính trọng số và tổng có trọng số
                for point in data:
                    dist = euclidean_distance(centroid, point)
                    weight = self.kernel_func(dist, self.bandwidth)
                    weights.append(weight)
                    weighted_sum += weight * point
                    weight_sum += weight
                
                # Tính centroid mới
                if weight_sum > 0:
                    new_centroid = weighted_sum / weight_sum
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(centroid)
            
            # Kiểm tra sự hội tụ
            new_centroids = np.array(new_centroids)
            shifts = np.linalg.norm(new_centroids - centroids, axis=1)
            centroids = new_centroids
            
            if np.max(shifts) < self.tol:
                print(f"\nĐã hội tụ sau {it+1} lần lặp")
                break
        
        # Tìm các tâm cụm duy nhất
        unique_centroids = []
        for centroid in centroids:
            if not any(np.linalg.norm(centroid - uc) < self.bandwidth / 2 for uc in unique_centroids):
                unique_centroids.append(centroid)
                
        self.cluster_centers_ = np.array(unique_centroids)
        print(f"Đã tìm thấy {len(self.cluster_centers_)} cụm")
        
        # Gán nhãn cụm cho từng điểm
        labels = []
        for point in data:
            distances = [euclidean_distance(point, center) for center in self.cluster_centers_]
            if len(distances) > 0:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                # Gán nhãn -1 cho các điểm nằm ngoài mọi cụm nếu cluster_all=False
                if not self.cluster_all and min_dist > self.bandwidth:
                    labels.append(-1)
                else:
                    labels.append(min_dist_idx)
            else:
                labels.append(-1)
                
        self.labels_ = np.array(labels)
        return self