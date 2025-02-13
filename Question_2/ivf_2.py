from sklearn.cluster import KMeans
import numpy as np

import torch

# vectors = np.random.rand(50, 128)
vectors = np.array(torch.load("data/train_embeddings.pth"))

from icecream import ic

# Step 1: Coarse Quantization (Clustering)
kmeans = KMeans(n_clusters=10)
centroids = kmeans.fit_predict(vectors)

# Step 2: Construct Posting Lists
posting_lists = {i: [] for i in range(10)}
for i, label in enumerate(centroids):
    posting_lists[label].append(i)

# Query Processing
# query = np.random.rand(128)
train_embeddings = np.array(torch.load("data/train_embeddings.pth"))

query = train_embeddings[0]

# print(query.shape)
# ic(query)

nearest_centroid = np.argmin([np.linalg.norm(query - centroid) for centroid in kmeans.cluster_centers_])

# ic(posting_lists)

# Fine Search (Within the posting list of nearest centroid)
nearest_vectors = [vectors[i] for i in posting_lists[nearest_centroid]]

print(nearest_vectors[:10])

closest_vector = min(nearest_vectors, key=lambda x: np.linalg.norm(query - x))

# print(f"Nearest vector: {closest_vector}")