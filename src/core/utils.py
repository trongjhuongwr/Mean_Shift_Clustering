import numpy as np

def euclidean_distance(point1, point2):
    """Tính khoảng cách Euclidean giữa hai điểm."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def flat_kernel(distance, bandwidth):
    """Hàm kernel phẳng: trả về 1 nếu khoảng cách < bandwidth, ngược lại 0."""
    return 1 if distance < bandwidth else 0

def gaussian_kernel(distance, bandwidth):
    """Hàm kernel Gaussian: trọng số giảm theo khoảng cách."""
    return np.exp(- (distance / bandwidth) ** 2)