import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

from icecream import ic

embeddings = torch.load("data/train_embeddings.pth") 
query_embeddings = torch.load("data/test_embeddings.pth") 

num_clusters = 10
nprobe_values = [1, 3, 5, 10] 

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(embeddings)
cluster_assignments = kmeans.predict(embeddings)

ivf_index = {i: [] for i in range(num_clusters)}
for idx, cluster_id in enumerate(cluster_assignments):
    ivf_index[cluster_id].append(embeddings[idx])

for key in ivf_index:
    ivf_index[key] = np.array(ivf_index[key])

def retrieve_images(query, nprobe, top_k=5):

    cluster_distances = cdist([query], kmeans.cluster_centers_)[0]
    closest_clusters = np.argsort(cluster_distances)[:nprobe]
    

    candidates = []
    for cluster_id in closest_clusters:
        if len(ivf_index[cluster_id]) > 0:
            candidates.extend(ivf_index[cluster_id])
    candidates = np.array(candidates)
    

    distances = cdist([query], candidates)[0]
    top_k_indices = np.argsort(distances)[:top_k]
    return top_k_indices, distances[top_k_indices]

comparisons_per_query = []
for nprobe in tqdm(nprobe_values, desc="Evaluating retrieval performance"):
    total_comparisons = 0
    for query in tqdm(query_embeddings, desc=f"Processing nprobe={nprobe}"):
        candidates_count = sum(len(ivf_index[c]) for c in np.argsort(cdist([query], kmeans.cluster_centers_)[0])[:nprobe])
        total_comparisons += candidates_count
    avg_comparisons = total_comparisons / len(query_embeddings)
    comparisons_per_query.append(avg_comparisons)

plt.figure(figsize=(10, 5))
plt.bar(range(num_clusters), [len(ivf_index[i]) for i in range(num_clusters)])
plt.xlabel("Cluster ID")
plt.ylabel("Number of Points")
plt.title("Number of Points in Each Cluster")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(nprobe_values, comparisons_per_query, marker='o')
plt.xlabel("nprobe")
plt.ylabel("Average Number of Comparisons")
plt.title("Comparisons vs. nprobe")
plt.show()

# ivf = IVF(k=10)
# ivf.fit(train_embeddings)

# retrieved_images, comparisons = ivf.image_to_image_retrieval(test_embeddings, top_k=10, nprobe=2)

# Step 6: Image to Image Retrieval Metrics
def mean_reciprocal_rank(results):
    ranks = []
    for i, res in enumerate(results):
        rank = np.where(np.isin(res, ground_truth[i]))[0]  # Find ranks of correct matches
        if len(rank) > 0:
            ranks.append(1 / (rank[0] + 1))  # Take the first correct match
        else:
            ranks.append(0)
    return np.mean(ranks)

def precision_at_k(results, k=100):
    return np.mean([len([x for x in res[:k] if x in ground_truth[i]]) / k for i, res in enumerate(results)])

def hit_rate(results, k=100):
    return np.mean([any(x in ground_truth[i] for x in res[:k]) for i, res in enumerate(results)])

query_size = 100 

ground_truth = torch.load("data/test_labels.pth")[:query_size]

# print(type(ground_truth))
# print(ground_truth.shape)
# print(ground_truth[:5])

from collections import Counter

labels_semantics = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

train_labels = torch.load("data/train_labels.pth").cpu().numpy()

def get_label_predictions(knn_values):
    index_to_labels = train_labels[knn_values]

    mapped = labels_semantics[np.array(index_to_labels)]

    most_common_labels = [Counter(row).most_common(1)[0][0] for row in mapped]
    
    return np.array(most_common_labels)

retrieval_results = np.array([retrieve_images(query, nprobe=30, top_k=5)[0] for query in tqdm(query_embeddings[:query_size], desc="Retrieving images")])
1

labelled_results = get_label_predictions(retrieval_results) # np.array([train_labels[np.array(i)] for i in retrieval_results])

labelled_gt = get_label_predictions(np.array(ground_truth).reshape(len(ground_truth), 1))

ic(ground_truth[:10])
ic(retrieval_results[0])
ic(np.array(retrieval_results).shape)   
ic(np.array(retrieval_results[0]).shape)
ic(labelled_gt)
ic(labelled_results)

ic(np.count_nonzero(labelled_gt == labelled_results))

mrr = mean_reciprocal_rank(retrieval_results)
precision = precision_at_k(retrieval_results)
hit_rate_score = hit_rate(retrieval_results)

print(f"Mean Reciprocal Rank: {mrr}")
print(f"Precision@100: {precision}")
print(f"Hit Rate: {hit_rate_score}")
