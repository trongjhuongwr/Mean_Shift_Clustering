import matplotlib.pyplot as plt

def plot_clusters(data, labels, centers=None):
    """Vẽ biểu đồ phân cụm cho hai đặc trưng đầu tiên."""
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    plt.title('Trực quan hóa phân cụm (income vs age)')
    plt.xlabel('Income')
    plt.ylabel('Age')
    plt.savefig('results/plots/cluster_plot.png')
    plt.show()