# Here I start to implement Decision Trees from scratch. 
# I hope to not take the word scratch literally because I would have to recreate the universe also.

from treenode import TreeNode

import numpy as np

class DecisionTree():
    def __init__():
        pass

    def entropy(self, class_probabilities):
        pass

    def calculate_information_gain(self, idk):
        pass

    def entropy_of_entire_data(self):
        pass

    def split(self):
        pass

    def find_best_split(self, data):
        pass

    def create_tree(self, data: np.array, current_depth: int):
        
        if current_depth > self.max_depth:
            return None
        
        split_1_data, split_2_data, split_feature_index, split_feature_val, split_entropy = self.find_best_split(data)

        label_probabilities = self.find_label_probs(data)

        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_entropy

        node = TreeNode(data, split_feature_index, split_feature_val, label_probabilities, information_gain)

        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node

        elif information_gain < self.min_information_gain:
            return node
        
        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)

        return node

    # basically tree banana
    def train(self, X_train: np.array, Y_train: np.array) -> None:
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        self.tree = self.create_tree(data=train_data, current_depth=0)

    