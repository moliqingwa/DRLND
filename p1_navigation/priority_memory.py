# -*- coding: utf-8 -*-#
# ----------------------------------------------------------------------
# Name:         memory
# Description:  
# Author:       zhenwang
# Date:         2019/12/18
# ----------------------------------------------------------------------
import random
from collections import deque

import numpy as np


class PrioritizedReplayBuffer(object):
    def __init__(self,
                 buffer_size=100000,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment_per_sampling=0.001,
                 epsilon=0.01,
                 abs_err_upper=1.,
                 min_prob_lower=1.e-6,
                 seed=0):
        self.buffer_size = buffer_size
        self.alpha = alpha  # [0~1] convert the importance of TD error to priority
        self.beta = beta  # importance-sampling, from initial experiences increasing to 1
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon   # small amount to avoid zero priority
        self.abs_err_upper = abs_err_upper  # clipped abs error
        self.min_prob_lower = min_prob_lower  # clipped min_prob

        self._count = 0

        self._tree = SumTree(buffer_size)
        # self._tree2 = PriorityTree(buffer_size)

        self._seed = np.random.seed(seed)

    def add(self, experiences):
        data_frame_priorities = self._tree.data_frame_priorities
        max_p = np.max(data_frame_priorities) if data_frame_priorities.any() else self.abs_err_upper

        self._tree.add(max_p, experiences)  # set the max p for new p
        '''
        max_p = self._tree2.max_probability(self.abs_err_upper)
        self._tree2.add(max_p, experiences)

        # normalize
        sum_priority = self._tree2.sum_priority()
        self._tree2.probability_array /= sum_priority
        '''
        self._count += 1

    def sample(self, batch_size):
        batch_idx, batch, weights = np.empty((batch_size,), dtype=np.int32), [], np.empty((batch_size, 1))

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        '''
        batch_idx = np.random.choice(len(self._tree2), batch_size,
                                     p=self._tree2.probability_array[0:len(self._tree2)])
        min_prob = self._tree2.min_probability(0)
        if min_prob < self.min_prob_lower:
            min_prob = self.min_prob_lower

        for j in range(batch_size):
            prob = self._tree2.probability_array[batch_idx[j]]
            weights[j, 0] = np.power(prob / min_prob, -self.beta)
            batch.append(self._tree2.data_array[batch_idx[j]])
        '''

        sum_priority = self._tree.total_priority
        priority_segment = sum_priority / batch_size  # priority segment

        min_prob = np.min(self._tree.data_frame_priorities) / sum_priority  # for later calculate weights
        if min_prob < self.min_prob_lower:
            min_prob = self.min_prob_lower

        for j in range(batch_size):
            lower, upper = priority_segment * j, priority_segment * (j + 1)
            v = np.random.uniform(lower, upper)
            idx, p, data = self._tree.get_leaf(v)
            prob = p / sum_priority
            # prob increase, weight decrease
            weights[j, 0] = np.power(prob / min_prob, -self.beta)
            batch_idx[j] = idx
            batch.append(data)

        return batch_idx, batch, weights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self._tree.update_leaf(ti, p)
            # self._tree2.probability_array[ti] = p

        # normalize
        # sum_priority = self._tree2.sum_priority()
        # self._tree2.probability_array /= sum_priority
        pass

    def __len__(self):
        return self._count if self._count < self.buffer_size else self.buffer_size


class PriorityTree(object):
    def __init__(self, capacity):
        self.capacity = capacity

        self.data_array = np.zeros(capacity, dtype=np.object)
        self.probability_array = np.zeros(capacity, dtype=np.float)

        self._write = 0
        self._is_full = False

    def max_probability(self, default_val_if_empty):
        return np.max(self.probability_array) if len(self) > 0 else default_val_if_empty

    def min_probability(self, default_val_if_empty):
        if len(self) == 0:
            return default_val_if_empty
        return np.min(self.probability_array) if self._is_full else np.min(self.probability_array[:self._write])

    def sum_priority(self):
        return np.sum(self.probability_array)

    def add(self, probability, data):
        self.data_array[self._write] = data
        self.probability_array[self._write] = probability

        self._write += 1
        if self._write >= self.capacity:
            self._write = 0
            self._is_full = True

    def __len__(self):
        return self._write if not self._is_full else self.capacity


class SumTree(object):
    """
    SumTree structure:
    [Parent nodes (size: capacity - 1) ... DataFrame nodes (size: capacity)]

        Parent nodes: indices range from 0 to (capacity - 2)
        DataFrame nodes: indices range from (capacity - 1) to (2 * capacity - 2)

    REFERENCE: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    """

    def __init__(self, capacity):
        """
        SumTree constructor.

        :param capacity: Specifies the size of data capacity (leaf nodes)
        """
        self.capacity = capacity
        self.write = 0
        self.is_full = False

        self._tree = np.zeros(2 * capacity - 1, dtype=float)
        self._data = np.zeros(capacity, dtype=object)

    @property
    def total_priority(self):
        """
        Retrieve total priority experiences.

        :return: Specifies total priority experiences.
        """
        return self._tree[0]

    @property
    def data_frame_priorities(self):
        """
        Retrieve data frame priorities.

        :return:
        """
        return self._tree[-self.capacity:] if self.is_full else \
            self._tree[-self.capacity: -self.capacity + self.write]

    def _propagate(self, idx, delta):
        """
        Propagate left node to parent node.

        :param idx: The left node index
        :param delta: The delta experiences to be added
        :return:
        """
        parent = (idx - 1) // 2
        self._tree[parent] += delta

        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx, val):
        """
        Retrieve child node index according to val.

        :param idx: Specifies the tree node index to search from.
        :param val: Specifies the priority experiences.

        :return: Leaf node index
        """
        left, right = 2 * idx + 1, 2 * idx + 2

        if left >= len(self._tree):  # leaf node
            return idx

        if val <= self._tree[left]:
            return self._retrieve(left, val)
        else:
            return self._retrieve(right, val - self._tree[left])

    def update_leaf(self, idx, priority):
        """
        Update tree index by priority

        :param idx: Specifies the tree node index to add
        :param priority: Specifies the priority experiences to add
        """
        delta = priority - self._tree[idx]

        self._tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority, data):
        """
        Add new data with priority.

        :param priority: the priority experiences of the data
        :param data: the data to be added
        """
        # leaf node index
        leaf_idx = self.write + self.capacity - 1
        # set data
        self._data[self.write] = data
        # update priority of the tree from leaf node (bottom upper)
        self.update_leaf(leaf_idx, priority)

        # move to next index
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.is_full = True

    def get_leaf(self, val):
        """
        Retrieve leaf info according to val.

        :param val: Specifies the experiences to retrieve

        :return: Returns leaf node index, leaf node priority and corresponding data experiences.
        """
        leaf_idx = self._retrieve(0, val)
        data_idx = leaf_idx - self.capacity + 1

        return leaf_idx, self._tree[leaf_idx], self._data[data_idx]


if __name__ == "__main__":
    tree = SumTree(4)
    tree.add(1, 3)
    tree.add(2, 4)
    tree.add(3, 5)
    tree.add(4, 6)
    # tree.add(6, 11)

    print(tree._tree, tree._data)
    print(tree.get_leaf(4))
