import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from scipy.spatial.distance import cdist

import numpy as np

class LSH:
    def __init__(self, num_hyperplanes, dim):
        self.num_hyperplanes = num_hyperplanes
        self.dim = dim
        self.hyperplanes = np.random.randn(num_hyperplanes, dim)
        self.buckets = dict()

    def hash_vector(self, vec):
        return tuple((vec @ self.hyperplanes.T) > 0)

    def insert(self, vec, index):
        hash_key = self.hash_vector(vec)
        if hash_key not in self.buckets:
            self.buckets[hash_key] = []
        self.buckets[hash_key].append(index)

    def query(self, query_vec, top_k=5):
        # Hash the query vector
        hash_key = self.hash_vector(query_vec)
        
        # Get the potential candidates from the same bucket
        candidates = self.buckets.get(hash_key, [])
        
        if len(candidates) == 0:
            return [], []

        # Compute the distances between the query vector and candidate vectors
        candidate_vectors = train_embeddings[candidates]  # Assuming train_embeddings is available
        distances = cdist([query_vec], candidate_vectors, metric='euclidean').flatten()
        
        # Get the indices of the top_k closest candidates
        top_k_indices = np.argsort(distances)[:top_k]
        
        # Return the top_k closest indices and their corresponding distances
        return [candidates[i] for i in top_k_indices], distances[top_k_indices]


test_embeddings = torch.load("data/test_embeddings.pth").cpu().numpy()
train_embeddings = torch.load("data/train_embeddings.pth").cpu().numpy()
train_labels = torch.load("data/train_labels.pth").cpu().numpy()
test_labels = torch.load("data/test_labels.pth").cpu().numpy()

n, d = train_embeddings.shape

query_vector = test_embeddings[0]  # Use the first test embedding as an example query

num_hyperplanes = 7  # Choose the number of hyperplanes
lsh = LSH(num_hyperplanes, d)

# Insert train embeddings
for idx, vec in tqdm(enumerate(train_embeddings), total = n, desc=f"LSH {num_hyperplanes}"):
    lsh.insert(vec, idx)

# Query the LSH structure
top_k_indices, top_k_distances = lsh.query(query_vector, top_k=5)

print("Top-5 closest vectors to the query:")
for idx, dist in zip(top_k_indices, top_k_distances):
    print(f"Index: {idx}, Distance: {dist}")

for i in top_k_indices:
    print(train_labels[i])


from collections import Counter

labels_semantics = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

train_labels = torch.load("data/train_labels.pth").cpu().numpy()

def get_label_predictions(knn_values):

    # index_to_labels = train_labels[knn_values]
    index_to_labels = train_labels[knn_values]

    mapped = labels_semantics[np.array(index_to_labels)]

    mapped = mapped.reshape(1, len(mapped))


    most_common_labels = [Counter(np.array(row).flatten()).most_common(1)[0][0] for row in mapped]
    
    # print("mcl", most_common_labels)
    return np.array(most_common_labels)

print(top_k_indices)

print(get_label_predictions(np.array(top_k_indices).reshape(len(top_k_indices), 1)))

print(labels_semantics[test_labels[0]])

total = 10000

final_label_values = []

for test_embedding in tqdm(test_embeddings[:total]):
    top_k_indices, top_k_distances = lsh.query(test_embedding, top_k=5)

    if len(top_k_distances) == 0:
        print("Haan zero wala mila")
        final_label_values.append([labels_semantics[0]])
        continue

    final_label_values.append(get_label_predictions(np.array(top_k_indices).reshape(len(top_k_indices), 1)))


def mean_reciprocal_rank(results, ground_truth):
    ranks = []
    for i, res in enumerate(results):
        rank = np.where(np.isin(res, ground_truth[i]))[0]  # Find ranks of correct matches
        if len(rank) > 0:
            ranks.append(1 / (rank[0] + 1))  # Take the first correct match
        else:
            ranks.append(0)
    return np.mean(ranks)

def precision_at_k(results, ground_truth, k=100):
    return np.mean([len([x for x in res[:k] if x in ground_truth[i]]) / k for i, res in enumerate(results)])

def hit_rate(results, ground_truth, k=100):
    return np.mean([any(x in ground_truth[i] for x in res[:k]) for i, res in enumerate(results)])

final_label_values = np.array(final_label_values).squeeze()

print(np.array(final_label_values).squeeze())
print(labels_semantics[test_labels[:total]])

def accuracy(predictions, labels):
    return np.sum(predictions == labels, axis = 0) / len(predictions)

acc = accuracy(final_label_values, labels_semantics[test_labels[:total]])

mrr = mean_reciprocal_rank(final_label_values, labels_semantics[test_labels[:total]])

pak = precision_at_k(final_label_values, labels_semantics[test_labels[:total]])

hr = hit_rate(final_label_values, labels_semantics[test_labels[:total]])

print("Accuracy:", acc * 100, "%")
print("Mean Reciprocal Rank:", mrr)
print("Precision at k:", pak)
print("Hit Rate:", hr)



# def take_mode

# n, d = train_embeddings.shape
# num_hyperplanes_list = [5, 10, 20, 50]

# # Create subplots for better visualization
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots
# axes = axes.flatten()  # Flatten for easier indexing

# for i, num_hyperplanes in enumerate(num_hyperplanes_list):
#     lsh = LSH(num_hyperplanes, d)

#     print(f"Inserting vectors with {num_hyperplanes} hyperplanes...")
#     for idx, vec in tqdm(enumerate(train_embeddings), total=n, desc=f"LSH {num_hyperplanes}"):
#         lsh.insert(vec, idx)

#     bucket_sizes = [len(v) for v in lsh.buckets.values()]

#     # Plot in corresponding subplot
#     ax = axes[i]
#     ax.hist(bucket_sizes, bins=20, alpha=0.7, label=f'{num_hyperplanes} hyperplanes')
#     ax.set_yscale("log")  # Log scale for better visibility
#     ax.set_title(f"LSH with {num_hyperplanes} Hyperplanes")
#     ax.set_xlabel("Bucket Size")
#     ax.set_ylabel("Frequency")
#     ax.legend()

# plt.tight_layout()
# plt.show()