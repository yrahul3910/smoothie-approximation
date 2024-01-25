import random

import numpy as np
from sklearn.neighbors import KDTree


class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left  = left
        self.right = right
        self.parent = None
        
        # The "support vectors"
        self.left_sv, self.right_sv = None, None
        if self.left is not None and self.right is not None:
            # Find the closest pair in left and right
            kdtree = KDTree(self.left.value)

            min_dist = np.inf
            for y in self.right.value:
                dist, ind = kdtree.query(y.reshape(1, -1), k=1)
                if dist < min_dist:
                    min_dist = dist
                    self.left_sv = self.left.value[ind[0]]
                    self.right_sv = y
            

class Fastmap:
    def __init__(self, x_train):
        self.x_train = x_train

    def _get_furthest_pair(self):
        # Pick a random point
        p1 = random.choice(self.x_train)

        # Find the point furthest from p1
        p2 = max(self.x_train, key=lambda x: np.linalg.norm(x - p1))

        # Find the point furthest from p2
        p3 = max(self.x_train, key=lambda x: np.linalg.norm(x - p2))

        return p2, p3

    def _split(self):
        # Split x_train based on the distance to the furthest pair
        p2, p3 = self._get_furthest_pair()

        x_train1 = np.array([x for x in self.x_train if np.linalg.norm(x - p2) < np.linalg.norm(x - p3)])
        x_train2 = np.array([x for x in self.x_train if np.linalg.norm(x - p2) >= np.linalg.norm(x - p3)])

        return x_train1, x_train2
    
    def _recurse(self):
        # Recurse until we have 10 points left
        if len(self.x_train) <= 10:
            return TreeNode(self.x_train, None, None)
        else:
            x_train1, x_train2 = self._split()
            tree_left = Fastmap(x_train1)._recurse()
            tree_right = Fastmap(x_train2)._recurse()

            new_node = TreeNode(self.x_train, tree_left, tree_right)
            new_node.left.parent = new_node
            new_node.right.parent = new_node

            return new_node

