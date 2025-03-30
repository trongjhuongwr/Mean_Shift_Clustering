import numpy as np
from .utils import euclidean_distance, flat_kernel, gaussian_kernel

class MeanShift:
    def __init__(self, bandwidth, kernel='flat', max_iter=100, tol=1e-5):
        """Khởi tạo mô hình Mean-Shift.
        
        Args:
            bandwidth (float): Bán kính để tính trung bình.
            kernel (str): Loại kernel ('flat' hoặc 'gaussian').
            max_iter (int): Số lần lặp tối đa.
            tol (float): Ngưỡng hội tụ.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        if kernel == 'flat':
            self.kernel_func = flat_kernel
        elif kernel == 'gaussian':
            self.kernel_func = gaussian_kernel
        else:
            raise ValueError("Kernel không được hỗ trợ")

    def fit(self, data):
        """Huấn luyện mô hình Mean-Shift trên dữ liệu.
        
        Args:
            data (np.ndarray): Dữ liệu đầu vào (n_samples, n_features).
        
        Returns:
            self: Đối tượng đã được huấn luyện.
        """
        centroids = data.copy()
        for it in range(self.max_iter):
            new_centroids = []
            for centroid in centroids:
                weights = []
                weighted_sum = np.zeros(data.shape[1])
                for point in data:
                    dist = euclidean_distance(centroid, point)
                    weight = self.kernel_func(dist, self.bandwidth)
                    weights.append(weight)
                    weighted_sum += weight * point
                new_centroid = weighted_sum / sum(weights)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)
            shifts = np.linalg.norm(new_centroids - centroids, axis=1)
            centroids = new_centroids
            if np.max(shifts) < self.tol:
                break
        
        # Tìm các tâm cụm duy nhất
        unique_centroids = []
        for centroid in centroids:
            if not any(np.linalg.norm(centroid - uc) < self.tol for uc in unique_centroids):
                unique_centroids.append(centroid)
        self.cluster_centers_ = np.array(unique_centroids)
        
        # Gán nhãn cụm
        labels = []
        for point in data:
            distances = [euclidean_distance(point, center) for center in self.cluster_centers_]
            label = np.argmin(distances)
            labels.append(label)
        self.labels_ = np.array(labels)
        return self