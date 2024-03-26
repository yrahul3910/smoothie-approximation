import random

import numpy as np


class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.parent = None


class Fastmap:
    def __init__(self, X, y):
        self.x = X
        self.y = y
        self.data = list(zip(self.x, self.y))

    def _get_furthest_pair(self):
        # Pick a random point
        p1 = random.choice(self.data)[0]

        # Find the point furthest from p1
        p2 = max(self.data, key=lambda x: np.linalg.norm(x[0] - p1))

        # Find the point furthest from p2
        p3 = max(self.data, key=lambda x: np.linalg.norm(x[0] - p2[0]))

        return p2, p3

    def _split(self):
        # Split x_train based on the distance to the furthest pair
        p2, p3 = self._get_furthest_pair()

        split1 = [x for x in self.data if np.linalg.norm(x[0] - p2[0]) < np.linalg.norm(x[0] - p3[0])]
        if len(split1) != 0:
            x1, y1 = zip(*split1)
        else:
            x1, y1 = [], []

        split2 = [x for x in self.data if np.linalg.norm(x[0] - p2[0]) >= np.linalg.norm(x[0] - p3[0])]
        if len(split2) != 0:
            x2, y2 = zip(*split2)
        else:
            x2, y2 = [], []

        return (x1, y1), (x2, y2)

    def _recurse(self):
        # Recurse until we have 10 points left
        if len(self.data) <= 10:
            return TreeNode(self.data, None, None)
        else:
            (x1, y1), (x2, y2) = self._split()
            if len(x1) == 0 or len(x2) == 0:
                return TreeNode(self.data, None, None)

            tree_left = Fastmap(np.array(x1), np.array(y1))._recurse()
            tree_right = Fastmap(np.array(x2), np.array(y2))._recurse()

            new_node = TreeNode(self.data, tree_left, tree_right)
            new_node.left.parent = new_node
            new_node.right.parent = new_node

            return new_node

