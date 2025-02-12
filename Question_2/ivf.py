import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import tqdm  # Import tqdm

import torch

class IVF:
    def __init__(self, k=10):
        self.k = k
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.inverted_index = {}
    
    def fit(self, data):

        self.kmeans.fit(data)
        labels = self.kmeans.labels_
        self.inverted_index = {i: [] for i in range(self.k)}
        
        for i, label in tqdm(enumerate(labels), total=len(labels), desc="Building Index"):
            self.inverted_index[label].append(data[i])
        
        for key in self.inverted_index:
            self.inverted_index[key] = np.array(self.inverted_index[key])
    
    def search(self, query, nprobe=1, top_k=5):

        centroids = self.kmeans.cluster_centers_
        dists = cdist([query], centroids, metric='euclidean')[0]
        closest_clusters = np.argsort(dists)[:nprobe]
        
        candidates = []
        comparisons = 0
        
        for cluster in closest_clusters:
            cluster_points = self.inverted_index.get(cluster, [])
            candidates.extend(cluster_points)
            comparisons += len(cluster_points)
        
        candidates = np.array(candidates)
        if len(candidates) == 0:
            return [], comparisons
        

        distances = cdist([query], candidates, metric='euclidean')[0]
        nearest_indices = np.argsort(distances)[:top_k]
        
        return [candidates[i] for i in nearest_indices], comparisons

    def image_to_image_retrieval(self, test_embeddings, top_k=100, nprobe=5):

        all_retrieved = []
        all_comparisons = []
        
        for test_embedding in tqdm(test_embeddings, desc="Searching"):  # Wrap with tqdm
            retrieved, comparisons = self.search(test_embedding, nprobe=nprobe, top_k=top_k)
            all_retrieved.append(retrieved)
            all_comparisons.append(comparisons)
        
        return all_retrieved, all_comparisons

    def plot_comparisons_vs_nprobe(self, query, max_nprobe=10, top_k=5):
        comparisons = []
        for nprobe in range(1, max_nprobe + 1):
            _, total_comparisons = self.search(query, nprobe=nprobe, top_k=top_k)
            comparisons.append(total_comparisons)
        
        plt.plot(range(1, max_nprobe + 1), comparisons)
        plt.xlabel('nprobe')
        plt.ylabel('Total Comparisons')
        plt.title('Comparisons vs. nprobe')
        plt.show()

    def plot_cluster_distribution(self):

        cluster_sizes = [len(self.inverted_index[i]) for i in range(self.k)]
        plt.bar(range(self.k), cluster_sizes)
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Points")
        plt.title("Cluster Distribution")
        plt.show()


test_embeddings = torch.load("data/test_embeddings.pth").cpu().numpy()
train_embeddings = torch.load("data/train_embeddings.pth").cpu().numpy()

ivf = IVF(k=10)
ivf.fit(train_embeddings)

retrieved_images, comparisons = ivf.image_to_image_retrieval(test_embeddings, top_k=10, nprobe=2)
