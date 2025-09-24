import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from distances import mahalanobis_dist_from_vectors


class SimilarityMatrix:
    def __init__(self, N):
        self.state_matrix = T.zeros((N + 1, N + 1))
        self.max_positive_distance = 0
        self.min_negative_distance = float("inf")

    def find_max_min_distances_in_batch(self, query_feature, gk_feature):
        dist = mahalanobis_dist_from_vectors(query_feature, gk_feature.reshape(1, -1))
        self.max_positive_distance = max(self.max_positive_distance, dist).reshape(1)
        self.min_negative_distance = min(self.min_negative_distance, dist).reshape(1)

    def reset_distances(self):
        self.max_positive_distance = 0
        self.min_negative_distance = float("inf")


class Agent(nn.Module):
    """
    RL agent with 3 fully connected layers.
    Input: flattened similarity matrix
    Output: action logits for selecting gallery images
    """

    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(Agent, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        # MLP
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.output = nn.Linear(fc3_dims, n_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

        # Device
        self.device = T.device("cuda" if T.cuda.is_available()
                               else "mps" if T.backends.mps.is_available()
                               else "cpu")
        self.to(self.device)

        # Similarity matrix tracker
        self.sim_mat = SimilarityMatrix(self.n_actions)

    def forward(self, state):
        """
        state: flattened similarity matrix
        """
        state = state.to(self.device)   # keep tensor type, don’t re-wrap
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.output(x)  # raw logits (not ReLU’d)
        return actions

    def compute_reward(self, label, margin=0.2):
        reward = margin + label * (self.sim_mat.max_positive_distance - self.sim_mat.min_negative_distance)
        return reward

    def update_state(self, state, label, g_k, threshold=0.4):
        if label:  # positive match
            z = (state[:, 0] + state[:, g_k]) / 2
            state[:, 0] = z
            state[0, :] = z
            state[0, g_k] = 1
            state[g_k, 0] = 1
        else:  # negative match
            z = state[:, g_k].detach().clone()
            z[z < threshold] = 0
            state[:, 0] = T.clamp(state[:, 0] - z, min=0)
            state[0, g_k] = 0
            state[g_k, 0] = 0

        z = state[:, 0]
        state[0, :] = z
        state.fill_diagonal_(0)
        return state

    def take_unique_action(self, logits, action_buffer):
        """
        Selects the max logit action not already taken.
        """
        max_logit = -float("inf")
        chosen_action = None
        for i in range(logits.shape[0]):
            if i not in action_buffer and logits[i] > max_logit:
                max_logit = logits[i]
                chosen_action = i
        return chosen_action
