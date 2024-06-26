import random
import torch
import numpy as np
import pickle

from replay_buffer.data_structures import SumSegmentTree


class ExperienceReplayMemory:
    '''
    Description:
        memory: 
            format example: [(s, a, r, s_, done, ...), (s, a, r, s_, done, ...), ...],
            each element in a transition should be a np.ndarray with 1D shape
    '''
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = [()] * capacity
        self.next_idx = 0
        self.is_full = False

    def push(self, transition: tuple[np.ndarray, ...]):
        self.memory[self.next_idx] = transition
        if self.next_idx + 1 == self.capacity:
            self.is_full = True
        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size: int):
        high = self.capacity - 1 if self.is_full else self.next_idx
        # sample_idx = random.sample(range(0, self.capacity if self.is_full else self.next_idx), batch_size)
        sample_idx = [random.randint(0, high) for _ in range(batch_size)]
        return *[np.vstack(x) for x in zip(*[self.memory[i] for i in sample_idx])], None, None

    def read_file(self, file_path: str):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.memory = data
        self.capacity = len(self.memory)
        self.is_full = True

    def read_data(self, data: list[tuple[np.ndarray, ...]]):
        '''
        each np.ndarray is a element of transition, e.g. state, action...
        '''
        self.memory = data
        self.capacity = len(data)
        self.is_full = True

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=20000, device=None):
        """Create Prioritized Replay buffer.
        Description
        ------------
        memory: 
            format example: [(s, a, r, s_, done, ...), (s, a, r, s_, done, ...), ...],
            each element in a transition should be a np.ndarray with 1D shape

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayMemory, self).__init__()
        self._storage = [()] * size
        self._maxsize = size
        self._next_idx = 0
        self._device = device
        self.is_full = False

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._max_priority = 100.0

    def read_file(self, file_path: str):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self._storage = data
        self._maxsize = len(self._storage)
        self.is_full = True
        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        for i in range(self._maxsize):
            self._it_sum[i] = self._max_priority ** self._alpha

    def read_data(self, data: list[tuple[np.ndarray, ...]]):
        '''
        each np.ndarray is a element of transition, e.g. state, action...
        shape should be one dimension
        '''
        self._storage = data
        self._maxsize = len(self._storage)
        self.is_full = True
        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        for i in range(self._maxsize):
            self._it_sum[i] = self._max_priority ** self._alpha

    def beta_by_frame(self, frame_idx: int) -> float:
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, data: tuple[np.ndarray, ...]):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        self._storage[self._next_idx] = data
        if self._next_idx + 1 == self._maxsize:
            self.is_full = True
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha


    def _encode_sample(self, idxes: list[int]):
        return [np.vstack(x) for x in zip(*[self._storage[i] for i in idxes])]

    def _sample_proportional(self, batch_size: int) -> list[int]:
        '''
            split to interval which has number of batch_size and get index in each of the interval,
            may have repeat sample
        '''
        res = list()
        s = self._it_sum.sum()
        for i in range(batch_size):
            mass = random.uniform(i / batch_size, (i + 1) / batch_size) * s
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size: int):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        idxes = self._sample_proportional(batch_size)

        weights = list()

        s = self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        for idx in idxes:
            p_sample = self._it_sum[idx] / s
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight)

        # max_weight use the smallest prob in the sample batch?
        max_weights = max(weights)
        weights = [weight / max_weights for weight in weights]
        weights = torch.tensor(weights, device=self._device, dtype=torch.float) 
        encoded_sample = self._encode_sample(idxes)
        return *encoded_sample, idxes, weights

    def update_priorities(self, idxes: list[int], priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            # assert 0 <= idx < len(self._storage)
            # assert (priority + 1e-8) < self._max_priority
            self._it_sum[idx] = (priority + 1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority+1e-5))