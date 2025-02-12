import numpy as np
import cv2
from scipy.spatial import distance

class SLIC_SELF_INITIALISE:
    def __init__(self, image, num_segments, compactness, clusters, keep_images = False):
        self.image = image

        self.num_segments = num_segments
        
        self.compactness = compactness

        self.height, self.width, self.channels = image.shape
        
        self.S = int(np.sqrt((self.height * self.width) / num_segments))
        
        self.clusters = clusters
        
        self.labels = -1 * np.ones((self.height, self.width), dtype=np.int32)
        
        self.distances = np.full((self.height, self.width), np.inf)

        self.keep_images = keep_images
        self.images = None

    def initialize_clusters(self):
        for y in range(self.S // 2, self.height, self.S):
            for x in range(self.S // 2, self.width, self.S):

                min_gradient = float('inf')
                best_y, best_x = y, x
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            grad = self.compute_gradient(ny, nx)
                            if grad < min_gradient:
                                min_gradient = grad
                                best_y, best_x = ny, nx
                color = self.image[best_y, best_x]
                self.clusters.append([best_x, best_y, *color])

    # Sobel Filter based gradient calculation (DIP flashbacks - 😭)
    def compute_gradient(self, y, x):
        dx = self.image[y, min(x + 1, self.width - 1)] - self.image[y, max(x - 1, 0)]
        dy = self.image[min(y + 1, self.height - 1), x] - self.image[max(y - 1, 0), x]
        return np.sum(dx**2 + dy**2)

    def update_clusters(self):
        for k, cluster in enumerate(self.clusters):
            cx, cy, *color = cluster
            for y in range(max(0, int(cy - self.S)), min(self.height, int(cy + self.S))):
                for x in range(max(0, int(cx - self.S)), min(self.width, int(cx + self.S))):
                    color_dist = np.linalg.norm(self.image[y, x] - color)
                    spatial_dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    # dist = np.sqrt((color_dist / self.compactness)**2 + (spatial_dist / self.S)**2)
                    
                    dist = color_dist + (self.compactness / self.S) * spatial_dist

                    if dist < self.distances[y, x]:
                        self.distances[y, x] = dist
                        self.labels[y, x] = k

    def update_clusters_vectorised(self):
        H, W, C = self.image.shape
        for k, cluster in enumerate(self.clusters):
            cx, cy, *color = cluster
            color = np.array(color)

            x_min, x_max = max(0, int(cx - self.S)), min(W, int(cx + self.S))
            y_min, y_max = max(0, int(cy - self.S)), min(H, int(cy + self.S))

            x_range = np.arange(x_min, x_max)
            y_range = np.arange(y_min, y_max)
            X, Y = np.meshgrid(x_range, y_range, indexing="xy")

            spatial_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

            color_dist = np.linalg.norm(self.image[y_min:y_max, x_min:x_max] - color, axis=2)

            dist = color_dist + (self.compactness / self.S) * spatial_dist

            mask = dist < self.distances[y_min:y_max, x_min:x_max]
            self.distances[y_min:y_max, x_min:x_max][mask] = dist[mask]
            self.labels[y_min:y_max, x_min:x_max][mask] = k

    def update_centers(self):
        new_clusters = []
        for k in range(len(self.clusters)):
            points = np.argwhere(self.labels == k)
            if len(points) == 0:
                new_clusters.append(self.clusters[k])
                continue
            colors = self.image[points[:, 0], points[:, 1]]
            new_center = [
                np.mean(points[:, 1]),  # x-coordinate
                np.mean(points[:, 0]),  # y-coordinate
                np.mean(colors[:, 0]),  # Red channel
                np.mean(colors[:, 1]),  # Green channel
                np.mean(colors[:, 2]),  # Blue channel
            ]
            new_clusters.append(new_center)
        self.clusters = new_clusters

    def enforce_connectivity(self):
        pass

    def iterate(self, max_iter=50, threshold=0.01):
        # self.initialize_clusters()

        for _ in range(max_iter):
            prev_clusters = np.array(self.clusters)
            self.update_clusters_vectorised()
    
            self.update_centers()

            if self.keep_images:
                segmentation = slic.get_segmentation()
                segmentation_bgr = cv2.cvtColor(segmentation, cv2.COLOR_LAB2BGR)
                cv2.imwrite(f"iters/segmentation{_}.jpg", segmentation_bgr)


            # Compute residual error
            residual_error = np.linalg.norm(prev_clusters - np.array(self.clusters))
            if residual_error < threshold:
                break


        self.enforce_connectivity()

    def get_segmentation(self):
        segmentation = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        for k, cluster in enumerate(self.clusters):
            segmentation[self.labels == k] = cluster[2:]  # Assign cluster color
        return segmentation


if __name__ == "__main__":
    # image = cv2.imread("data/frame_0000.jpg")
    image = cv2.imread("/home/anshium/workspace/courses/smai/smai-assignment-1/Question_5/more_images/SLIC/2.jpg")
    # image = cv2.imread("/home/anshium/Pictures/wallpapers/Fantasy-Lake2.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    slic = SLIC(image, num_segments=100, compactness=20.0)

    slic.iterate()
    segmentation = slic.get_segmentation()

    # Convert LAB back to BGR and save
    segmentation_bgr = cv2.cvtColor(segmentation, cv2.COLOR_LAB2BGR)
    cv2.imwrite("segmentation.jpg", segmentation_bgr)
