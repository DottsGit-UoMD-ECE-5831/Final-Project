#Part 2  without sklearn ----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(42)

# Define function to load the Iris dataset
def load_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    data = np.genfromtxt(url, delimiter=',', usecols=range(4), dtype=float)
    target = np.genfromtxt(url, delimiter=',', usecols=4, dtype=str)
    target = np.where(target == 'Iris-setosa', 0, np.where(target == 'Iris-versicolor', 1, 2))
    return data, target, feature_names

# Define function to standardize features
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K=5, max_iters=100, feature_index_1=0, feature_index_2=1):
        self.K = K
        self.max_iters = max_iters
        self.feature_index_1 = feature_index_1
        self.feature_index_2 = feature_index_2

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        return self._assign_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_mean = np.mean(self.X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return np.all(np.array(distances) < 1e-6)

    def _assign_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def plot(self, true_labels, title_suffix=''):
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(self.X[:, self.feature_index_1], self.X[:, self.feature_index_2],
                            c=true_labels, cmap='viridis', s=50, alpha=0.6)

        for i, index in enumerate(self.clusters):
            point = [self.X[index].T[self.feature_index_1], self.X[index].T[self.feature_index_2]]
            ax.scatter(*point, label=f'Cluster {i+1}', edgecolor='k', s=30)

        for c in self.centroids:
            point = [c[self.feature_index_1], c[self.feature_index_2]]
            ax.scatter(*point, marker="x", color="red", s=100, linewidth=2, label='Centroids')

        ax.set_title(f'KMeans Clustering Results ({title_suffix})')
        ax.set_xlabel(f'{self.feature_names[self.feature_index_1]}')
        ax.set_ylabel(f'{self.feature_names[self.feature_index_2]}')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='True Label')
        plt.show()

def plot_elbow_curve(X, k_range):
    inertia_values = []
    for k in k_range:
        kmeans = KMeans(K=k, max_iters=150)
        inertia = kmeans.predict(X)
        inertia_values.append(inertia)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve for Optimal K')
    plt.grid(True)
    plt.show()

def plot_clusters_for_optimal_k(X, y, optimal_k, feature_indices):
    kmeans = KMeans(K=optimal_k, max_iters=150, feature_index_1=feature_indices[0], feature_index_2=feature_indices[1])
    kmeans.feature_names = feature_names
    cluster_labels = kmeans.predict(X)
    
    # Calculate confusion matrix
    confusion = confusion_matrix(y, cluster_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(optimal_k), yticklabels=np.arange(3))
    plt.xlabel('Cluster Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for K={optimal_k}')
    plt.show()

    # Plot clustering results
    for i in range(10):  # Plot 10 iterations for the optimal K
        kmeans = KMeans(K=optimal_k, max_iters=150, feature_index_1=feature_indices[0], feature_index_2=feature_indices[1])
        kmeans.feature_names = feature_names
        kmeans.predict(X)
        kmeans.plot(true_labels=y, title_suffix=f'K={optimal_k} Iteration={i+1}')

# Testing with Iris Dataset
if __name__ == "__main__":
    # Load Iris dataset
    X, y, feature_names = load_iris()

    # Standardize features
    X_scaled = standardize(X)

    # Determine the optimal number of clusters using the elbow method
    k_range = range(1, 11)
    plot_elbow_curve(X_scaled, k_range)

    # For demonstration, assume the optimal K from elbow plot is 3
    optimal_k = 3
    feature_indices = (2, 3)  # Example: Petal length and petal width

    # Plot clusters for each of the 10 iterations for the optimal K
    plot_clusters_for_optimal_k(X_scaled, y, optimal_k, feature_indices)
