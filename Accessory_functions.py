def estimate_dataset_size(dataset):
    # Get one sample to compute size
    sample, _ = dataset[0]
    
    # Calculate memory size of the sample
    sample_size_bytes = sample.element_size() * sample.numel()

    # Estimate total dataset size
    num_samples = len(dataset)
    total_size_bytes = sample_size_bytes * num_samples

    # Convert to GB
    total_size_gb = total_size_bytes / (1024 ** 3)
    
    return total_size_gb


import matplotlib.pyplot as plt
from collections import Counter

def get_label_distribution(dataset):
    labels = [label for _, label in dataset]  # Assuming (data, label) format
    label_counts = Counter(labels)
    return label_counts

label_mapping = {0: "Und.", 1: "Low", 2: "Med", 3: "High"}

# train_label_counts = get_label_distribution(train_dataset)
# val_label_counts = get_label_distribution(val_dataset)
# test_label_counts = get_label_distribution(test_dataset)

def plot_label_distribution(label_counts, title):
    labels, counts = zip(*sorted(label_counts.items()))  
    custom_labels = [label_mapping[label] for label in labels] 
    
    plt.bar(custom_labels, counts)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_and_average_images(directory):
    time_series = []
    
    # Loop through all .pt files in the directory
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            file_path = os.path.join(directory, file)
            data = torch.load(file_path)  # Load the tensor
            
            if data.shape != (1, 180, 500, 500):
                raise ValueError(f"Unexpected shape {data.shape} in file {file}")
            
            data = data.squeeze(0)  # Remove the grayscale channel to get [180, 500, 500]
            avg_brightness = data.mean(dim=(1, 2))  # Average over height and width
            time_series.append(avg_brightness.numpy())
    
    # Convert to numpy array and average over all files
    time_series = np.array(time_series)  # Shape: [num_files, 180]
    avg_over_files = np.mean(time_series, axis=0)  # Average across files
    
    return avg_over_files

def plot_brightness(time_series_avg):
    plt.figure(figsize=(8, 5))
    plt.plot(time_series_avg, marker='o', linestyle='-', label='Avg Brightness')
    plt.xlabel('Time Frame')
    plt.ylabel('Average Brightness')
    plt.title('Brightness Change Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def export_to_csv(time_series_avg, output_path):
    df = pd.DataFrame({'Frame': np.arange(len(time_series_avg)), 'Average Intensity': time_series_avg})
    df.to_csv(output_path, index=False)

# Set directory path
directory = r"./../../Datasets/interpolated_torch_tensors/"  # Change this to your actual directory path
output_csv = r"./../../Datasets/average_brightness.csv"  # Output CSV file name

# Load and process images
time_series_avg = load_and_average_images(directory)

# Plot results
plot_brightness(time_series_avg)

# Export to CSV
export_to_csv(time_series_avg, output_csv)

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'../../Datasets/average_brightness.csv')
plt.plot(data['Frame'], data['Average Intensity'])

heuristic_1 = [20, 69, 89, 109, 129, 149, 179]
New = [20, 49, 69, 89, 109, 129, 149, 179]

plt.vlines(heuristic_1, ymin=0, ymax=data['Average Intensity'].max(), color='orange', label='Heuristic')
plt.vlines(New, ymin=0, ymax=data['Average Intensity'].max(), color='m', linestyle='dashed', linewidth=3, alpha=0.8, label='New')

plt.legend()
plt.xlabel('Frame (Time)')
plt.ylabel('Intensity Average over All Images')
