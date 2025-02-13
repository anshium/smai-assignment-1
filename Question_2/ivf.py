import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

train_embeddings = torch.load('data/train_embeddings.pth').cpu().numpy()  # shape: (N_train, d)
test_embeddings  = torch.load('data/test_embeddings.pth').cpu().numpy()   # shape: (N_test, d)
train_labels     = torch.load('data/train_labels.pth').cpu().numpy()        # shape: (N_train,)
test_labels      = torch.load('data/test_labels.pth').cpu().numpy()         # shape: (N_test,)

text_embeddings  = torch.load('data/text_embedding.pth').cpu().numpy()      # shape: (10, d)


def build_ivf_index(embeddings, n_clusters=10):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    centroids = kmeans.cluster_centers_
    assignments = kmeans.labels_
    inverted_index = {}
    for cid in range(n_clusters):
        inverted_index[cid] = np.where(assignments == cid)[0]
    return centroids, inverted_index, assignments, kmeans

def ivf_query(query, train_embs, centroids, inverted_index, nprobe=1, metric='euclidean', k=100):

    if not isinstance(query, np.ndarray):
        query = query.detach().cpu().numpy()
    if metric == 'euclidean':
        centroid_dists = np.linalg.norm(centroids - query, axis=1)
    elif metric == 'cosine':
        q_norm = np.linalg.norm(query)
        centroid_norms = np.linalg.norm(centroids, axis=1)
        centroid_dists = 1 - np.dot(centroids, query) / (centroid_norms * q_norm + 1e-8)
    sorted_clusters = np.argsort(centroid_dists)
    candidate_indices = []
    comparisons = 0
    for cluster in sorted_clusters[:nprobe]:
        indices = inverted_index[cluster]
        candidate_indices.extend(indices.tolist())
        comparisons += len(indices)
    candidate_embs = train_embs[candidate_indices]
    if metric == 'euclidean':
        dists = np.linalg.norm(candidate_embs - query, axis=1)
    elif metric == 'cosine':
        q_norm = np.linalg.norm(query)
        cand_norms = np.linalg.norm(candidate_embs, axis=1)
        dists = 1 - np.dot(candidate_embs, query) / (cand_norms * q_norm + 1e-8)
    sorted_idx = np.argsort(dists)[:k]
    retrieved_indices = np.array(candidate_indices)[sorted_idx]
    return retrieved_indices, comparisons

def compute_ivf_retrieval_metrics(query_embs, query_labels, train_embs, train_labels, centroids, inverted_index, nprobe, k=100, metric='euclidean'):
    mrrs, precisions, hits, total_comparisons = [], [], [], []
    for i, query in enumerate(query_embs):
        indices, comparisons = ivf_query(query, train_embs, centroids, inverted_index, nprobe=nprobe, metric=metric, k=k)
        total_comparisons.append(comparisons)
        retrieved_labels = train_labels[indices]
        true_label = query_labels[i]
        correct_positions = np.where(retrieved_labels == true_label)[0]
        mrrs.append(1.0 / (correct_positions[0] + 1) if len(correct_positions) > 0 else 0)
        precisions.append(np.sum(retrieved_labels == true_label) / float(k))
        hits.append(1 if np.any(retrieved_labels == true_label) else 0)
    return np.mean(mrrs), np.mean(precisions), np.mean(hits), np.mean(total_comparisons)


if __name__ == '__main__':
    n_clusters = 10
    centroids, inverted_index, assignments, kmeans_model = build_ivf_index(train_embeddings, n_clusters=n_clusters)
    
    cluster_sizes = [len(inverted_index[i]) for i in range(n_clusters)]
    plt.figure()
    plt.bar(range(n_clusters), cluster_sizes, color='skyblue')
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Points")
    plt.title("IVF: Points per Cluster")
    plt.show()
    
    ivf_results = {}
    for nprobe in [1, 2, 5, 10]:
        mrr_ivf, prec_ivf, hit_ivf, comp_ivf = compute_ivf_retrieval_metrics(
            test_embeddings, test_labels, train_embeddings, train_labels, centroids, inverted_index,
            nprobe=nprobe, k=100, metric='euclidean')
        ivf_results[nprobe] = {'MRR': mrr_ivf, 'Precision': prec_ivf, 'HitRate': hit_ivf, 'AvgComparisons': comp_ivf}
        print(f"IVF (nprobe={nprobe}): MRR = {mrr_ivf:.4f}, Precision@100 = {prec_ivf:.4f}, Hit Rate = {hit_ivf:.4f}, Avg Comparisons = {comp_ivf:.2f}")
    
    avg_comps_ivf = [ivf_results[np]['AvgComparisons'] for np in [1, 2, 5, 10]]
    plt.figure()
    plt.plot([1, 2, 5, 10], avg_comps_ivf, marker='o', linestyle='-')
    plt.xlabel("nprobe")
    plt.ylabel("Avg Comparisons per Query")
    plt.title("IVF: Avg Comparisons vs. nprobe")
    plt.grid(True)
    plt.show()