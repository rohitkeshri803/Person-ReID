from agent import Agent
from data_loader import Market1501, load_model
from feature_extractor import model
from distances import calculate_similarity

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms
import os

# ===== Device handling =====
device = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

if T.cuda.is_available():  # safer than set_default_tensor_type
    T.set_default_tensor_type(T.cuda.FloatTensor)

# ===== Hyperparameters =====
Ns = 30
input_dims = (Ns + 1) * (Ns + 1)
fc1_dims = 256
fc2_dims = 256
fc3_dims = 256
GAMMA = 0.9
max_epoch = 30
Kmax = 10

# ===== Initialize agent & feature extractor =====
agent = Agent(
    ALPHA=0.001,
    input_dims=input_dims,
    fc1_dims=fc1_dims,
    fc2_dims=fc2_dims,
    fc3_dims=fc3_dims,
    n_actions=Ns,
).to(device)

feature_extractor = model().to(device)
market = Market1501()

tau_p = []
print("==> Start training")

for epoch in range(max_epoch):

    expected_return = 0
    action_buffer = []

    query, query_label, g, g_labels = market.nextbatch(n=30)

    # Move to device
    query, query_label = query.to(device), query_label.to(device)
    g, g_labels = g.to(device), g_labels.to(device)

    g_features = feature_extractor(g)
    query_feature = feature_extractor(query)

    state = calculate_similarity(T.cat((query_feature, g_features)))

    for t in range(Kmax):
        logits = agent(state.flatten())
        action = agent.take_unique_action(logits, action_buffer)
        action_buffer.append(action)

        gk_feature = g_features[action]
        g_label = g_labels[action]

        agent.sim_mat.find_max_min_distances_in_batch(query_feature, gk_feature)

        label = 1 if g_label == query_label else -1
        state = agent.update_state(state, label, action + 1)
        agent.sim_mat.state_matrix = state
        tau_p.append((query_feature, gk_feature, label))

        reward = agent.compute_reward(label)
        expected_return -= GAMMA * reward

    agent.optimizer.zero_grad()
    expected_return.backward()
    agent.optimizer.step()
    agent.sim_mat.reset_distances()

    print(f"epoch {epoch+1}/{max_epoch}\t Expected return: {expected_return.item():.4f}")

    # ===== Save checkpoint after each epoch =====
    ckpt_path = f"checkpoint_epoch{epoch+1}.pth"
    T.save({
        "agent_state": agent.state_dict(),
        "feature_extractor_state": feature_extractor.state_dict(),
        "epoch": epoch + 1,
    }, ckpt_path)
    print(f"[INFO] Saved checkpoint -> {ckpt_path}")

# ===== Save final model =====
final_path = "my_model.pth"
T.save({
    "agent_state": agent.state_dict(),
    "feature_extractor_state": feature_extractor.state_dict(),
    "epoch": max_epoch,
}, final_path)
print(f"[INFO] Training complete. Final model saved as {final_path}")
