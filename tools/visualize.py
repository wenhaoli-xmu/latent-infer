import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, action='append', required=True)
args = parser.parse_args()

# Collect data from all paths
data_list = []
for path in args.path:
    with open(path, 'r') as f:
        data = json.load(f)
    loss = data['loss']
    baseline = data['baseline']
    if len(loss) != len(baseline):
        raise ValueError("Length of 'loss' and 'baseline' must be the same.")
    loss_minus_baseline = [l - b for l, b in zip(loss, baseline)]
    data_list.append(loss_minus_baseline)

# Check if all datasets have the same length
lengths = [len(d) for d in data_list]
if not all(l == lengths[0] for l in lengths):
    raise ValueError("All datasets must have the same number of steps.")

steps = range(len(data_list[0]))

def average_filter(data, window):
    window = int(window)
    return np.convolve(data, np.ones(window)/window, mode='same')

# Create figure and subplots
plt.figure(figsize=(15,10))

# Colors for each path
colors = plt.cm.tab10.colors

# Left top: Original curves
plt.subplot(2,2,1)
for i, data in enumerate(data_list):
    plt.plot(steps, data, label=f'Path {i+1}', color=colors[i])
plt.title('No Average Filter')
plt.xlabel('Step')
plt.ylabel('Loss - Baseline')
plt.legend()
plt.grid(True)

# Left bottom: Window=4
plt.subplot(2,2,3)
for i, data in enumerate(data_list):
    filtered = average_filter(data, 4)
    plt.plot(steps, filtered, label=f'Path {i+1}', color=colors[i])
plt.title('Average Filter, Window=4')
plt.xlabel('Step')
plt.ylabel('Loss - Baseline')
plt.legend()
plt.grid(True)

# Middle: Window=16
plt.subplot(2,2,2)
for i, data in enumerate(data_list):
    filtered = average_filter(data, 16)
    plt.plot(steps, filtered, label=f'Path {i+1}', color=colors[i])
plt.title('Average Filter, Window=16')
plt.xlabel('Step')
plt.ylabel('Loss - Baseline')
plt.legend()
plt.grid(True)

# Right bottom: Window=64
plt.subplot(2,2,4)
for i, data in enumerate(data_list):
    filtered = average_filter(data, 64)
    plt.plot(steps, filtered, label=f'Path {i+1}', color=colors[i])
plt.title('Average Filter, Window=64')
plt.xlabel('Step')
plt.ylabel('Loss - Baseline')
plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the figure
filename = "combined_plot.jpg"
plt.savefig(f"vis_{filename}")

# Close the plot
plt.close()