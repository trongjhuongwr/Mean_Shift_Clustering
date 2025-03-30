from sklearn.metrics import silhouette_score, davies_bouldin_score

def calculate_silhouette_score(data, labels):
    """Tính Silhouette Score."""
    return silhouette_score(data, labels)

def calculate_davies_bouldin_score(data, labels):
    """Tính Davies-Bouldin Index."""
    return davies_bouldin_score(data, labels)