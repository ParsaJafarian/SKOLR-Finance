"""Plot time series sample predictions from the test set."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description="Plot time series predictions")
parser.add_argument('--results_path', type=str, required=True, help="Path to results folder containing x.npy, pred.npy, true.npy")
args = parser.parse_args()

BASE_PATH = "results/" + args.results_path

# Load files
x = np.load(os.path.join(BASE_PATH, "x.npy"))
preds = np.load(os.path.join(BASE_PATH, "pred.npy"))
trues = np.load(os.path.join(BASE_PATH, "true.npy"))

print(f"x shape: {x.shape}")
print(f"preds shape: {preds.shape}")
print(f"trues shape: {trues.shape}")

feature_idx = 0
num_samples_to_plot = 9
seq_len = x.shape[1]

fig, axes = plt.subplots(3, 3, figsize=(15, 9))
axes = axes.flatten()

for i in range(num_samples_to_plot):
    ax = axes[i]

    x_series = x[seq_len * i, :, feature_idx]
    true_series = trues[seq_len * i, :, feature_idx]
    pred_series = preds[seq_len * i, :, feature_idx]

    seq_len = x_series.shape[0]
    pred_len = true_series.shape[0]

    t_x = np.arange(seq_len)
    t_y = np.arange(seq_len, seq_len + pred_len)

    ax.plot(t_x, x_series, label="Input")
    ax.plot(t_y, true_series, label="True")
    ax.plot(t_y, pred_series, label="Pred")

    ax.axvline(seq_len - 1, linestyle="--", linewidth=0.8)
    ax.set_title(f"Sample {i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

# Legend only once
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

plt.tight_layout()
plt.show()
