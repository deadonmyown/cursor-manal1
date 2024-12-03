import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

def d(j: list[float]) -> int:
    ds = [float('inf')] * len(j)
    for i in range(1, len(j)-1):
        ds[i] = abs(j[i]-j[i+1]) / abs(j[i-1]-j[i])
    
    m = min(ds)
    for i in range(len(ds)):
        if ds[i] == m:
            return i

def find_optimal_clusters_count(flowersData):
    from sklearn.cluster import KMeans
    
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(flowersData)
        inertia.append(kmeans.inertia_)

    optimal_count = d(inertia)+1
    return optimal_count

class KMeans:

    def __init__(self, clusters_count=8, max_iterations=300, epsilon=1e-8, random_seed=None):
        setattr(self, 'clusters_count', clusters_count)
        setattr(self, 'max_iterations', max_iterations)
        setattr(self, 'epsilon', epsilon)

        self._centroids = None
        self.random_seed = random_seed

    @staticmethod
    def _distances_from_expanded_data_and_centroids(expanded_data, centroids, features_axis):
        return np.linalg.norm(expanded_data - centroids, axis=features_axis)

    @staticmethod
    def _supposed_clusters_by(data_with_distances, distances_axis):
        return np.argmin(data_with_distances, axis=distances_axis)

    @property
    def centroids(self):
        return self._centroids

    def train(self, X: np.array, provide_intermediate_centroids=None) -> None:
        data = np.unique(X, axis=0)
        n, features_count = data.shape

        rnd = np.random.default_rng(seed=self.random_seed)
        centroids = data[rnd.choice(n, self.clusters_count, replace=False)]

        if provide_intermediate_centroids is not None:
            provide_intermediate_centroids(data, centroids)

        expanded_data = np.reshape(data, (n, 1, features_count))
        distances_axis = 1
        features_axis = 2

        for iteration in range(self.max_iterations):
            data_with_distances = KMeans\
                ._distances_from_expanded_data_and_centroids(expanded_data, centroids, features_axis)
            data_supposed_cluster = KMeans._supposed_clusters_by(data_with_distances, distances_axis)

            new_centroids = np.array([data[data_supposed_cluster == i].mean(axis=0) for i in range(self.clusters_count)])

            if np.linalg.norm(new_centroids - centroids) <= self.epsilon:
                break

            centroids = new_centroids

            if provide_intermediate_centroids is not None:
                provide_intermediate_centroids(data, centroids, labels=data_supposed_cluster)

        self._centroids = centroids

        return data_supposed_cluster

    def predict(self, X: np.array) -> np.array:
        distances_axis = 1
        features_axis = 2
        data_with_distances = KMeans \
            ._distances_from_expanded_data_and_centroids(X[:, np.newaxis], self._centroids, features_axis)
        return KMeans._supposed_clusters_by(data_with_distances, distances_axis)


if __name__ == "__main__":
    flowers = load_iris()
    data = flowers['data']
    targets = flowers['target']

    clusters_count = find_optimal_clusters_count(data)
    kMeans = KMeans(clusters_count=clusters_count)

    def visualize_training(actual_data, centroids, labels=None):
        fig, ax = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                ax_plt = ax[i][j]
                ax_plt.scatter(actual_data[:, i], actual_data[:, j], c=labels)
                ax_plt.scatter(centroids[:, i], centroids[:, j], color="red", marker="x")

        plt.show()

    kMeans.train(data, visualize_training)

