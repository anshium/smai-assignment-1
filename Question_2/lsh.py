import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

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

test_embeddings = torch.load("data/test_embeddings.pth").cpu().numpy()
train_embeddings = torch.load("data/train_embeddings.pth").cpu().numpy()

n, d = train_embeddings.shape
num_hyperplanes_list = [5, 10, 20, 50]

# Create subplots for better visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots
axes = axes.flatten()  # Flatten for easier indexing

for i, num_hyperplanes in enumerate(num_hyperplanes_list):
    lsh = LSH(num_hyperplanes, d)

    print(f"Inserting vectors with {num_hyperplanes} hyperplanes...")
    for idx, vec in tqdm(enumerate(train_embeddings), total=n, desc=f"LSH {num_hyperplanes}"):
        lsh.insert(vec, idx)

    bucket_sizes = [len(v) for v in lsh.buckets.values()]

    # Plot in corresponding subplot
    ax = axes[i]
    ax.hist(bucket_sizes, bins=20, alpha=0.7, label=f'{num_hyperplanes} hyperplanes')
    ax.set_yscale("log")  # Log scale for better visibility
    ax.set_title(f"LSH with {num_hyperplanes} Hyperplanes")
    ax.set_xlabel("Bucket Size")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.show()
