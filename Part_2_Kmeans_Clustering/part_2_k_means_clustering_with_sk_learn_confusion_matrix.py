
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans as SK_KMeans
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Define function to load the Iris dataset
def load_iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names

# Define function to standardize features
def standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Define the KMeans class from scratch
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
        distances = [np.linalg.norm(sample - point) for point in centroids]
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
        distances = [np.linalg.norm(centroids_old[i] - centroids[i]) for i in range(self.K)]
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
        ax.set_xlabel(f'{feature_names[self.feature_index_1]}')
        ax.set_ylabel(f'{feature_names[self.feature_index_2]}')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='True Label')
        plt.show()

def plot_elbow_curve(X, k_range):
    inertia_values = []
    for k in k_range:
        kmeans = KMeans(K=k, max_iters=150)
        kmeans.predict(X)
        # Calculate inertia as the sum of squared distances from samples to their closest centroid
        inertia = np.sum([np.linalg.norm(X[i] - kmeans.centroids[kmeans._closest_centroid(X[i], kmeans.centroids)])**2 for i in range(X.shape[0])])
        inertia_values.append(inertia)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve for Optimal K')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

# Testing with Iris Dataset
if __name__ == "__main__":
    # Load and standardize Iris dataset
    X, y, feature_names = load_iris_data()
    X_scaled = standardize(X)

    # Determine the optimal number of clusters using the elbow method
    k_range = range(1, 11)
    plot_elbow_curve(X_scaled, k_range)

    # For demonstration, assume the optimal K from elbow plot is 3
    optimal_k = 3
    feature_indices = (2, 3)  # Example: Petal length and petal width

    # Custom KMeans clustering
    custom_kmeans = KMeans(K=optimal_k, max_iters=150, feature_index_1=feature_indices[0], feature_index_2=feature_indices[1])
    custom_labels = custom_kmeans.predict(X_scaled)

    # Scikit-Learn KMeans clustering
    sk_kmeans = SK_KMeans(n_clusters=optimal_k, max_iter=150, random_state=42)
    sk_labels = sk_kmeans.fit_predict(X_scaled)

    # Plot confusion matrices for custom and Scikit-Learn KMeans
    plot_confusion_matrix(y, custom_labels, title='Custom KMeans Confusion Matrix')
    plot_confusion_matrix(y, sk_labels, title='Scikit-Learn KMeans Confusion Matrix')

    # Plot clusters for both methods
    custom_kmeans.plot(true_labels=y, title_suffix=f'Custom KMeans K={optimal_k}')
    plt.figure(figsize=(12, 8))
    plt.scatter(X_scaled[:, feature_indices[0]], X_scaled[:, feature_indices[1]], c=sk_labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(sk_kmeans.cluster_centers_[:, feature_indices[0]], sk_kmeans.cluster_centers_[:, feature_indices[1]], marker='x', color='red', s=100, linewidth=2, label='Centroids')
    plt.xlabel(feature_names[feature_indices[0]])
    plt.ylabel(feature_names[feature_indices[1]])
    plt.title(f'Scikit-Learn KMeans Clustering Results (K={optimal_k})')
    plt.legend()
    plt.colorbar(label='Cluster Label')
    plt.show()
