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

import os
import torch
import torch.nn.functional as F

def process_and_save(root_dir, save_dir, target_size=(500, 500)):
    """
    Extracts the first 20 frames' average and 6 selected frames, resizes, and saves to new .pt files.
    """
    classes = ['undetectable', 'low', 'medium', 'high']
    selected_frame_indices = [69, 89, 109, 129, 149, 179]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for label in classes:
        class_path = os.path.join(root_dir, label)
        save_class_path = os.path.join(save_dir, label)
        
        if not os.path.exists(class_path):
            continue  # Skip if folder doesn't exist
        
        if not os.path.exists(save_class_path):
            os.makedirs(save_class_path)
        
        for file in os.listdir(class_path):
            if file.endswith('.pt'):
                file_path = os.path.join(class_path, file)
                save_file_path = os.path.join(save_class_path, file)
                
                # Load tensor
                tensor_data = torch.load(file_path, map_location='cpu')  # [C, T, H, W]
                max_frames = tensor_data.shape[1]
                
                # Ensure enough frames
                valid_frames = [i for i in selected_frame_indices if i < max_frames]
                if len(valid_frames) < 6:
                    print(f"Skipping {file_path}: Not enough frames ({max_frames} available, required 180)")
                    continue
                
                # Compute average of the first 20 frames
                avg_first_20 = torch.mean(tensor_data[:, :20, :, :], dim=1, keepdim=True)  # [C, 1, H, W]
                selected_frames = tensor_data[:, valid_frames, :, :]  # [C, 6, H, W]
                
                # Concatenate to form a 7-frame tensor
                final_tensor = torch.cat((avg_first_20, selected_frames), dim=1)  # [C, 7, H, W]
                final_tensor = final_tensor.squeeze(0) if final_tensor.shape[0] == 1 else final_tensor
                
                # Resize
                if final_tensor.dim() == 3:
                    final_tensor = final_tensor.unsqueeze(0)  # -> [1, 7, H, W]
                
                resized_tensor = F.interpolate(
                    final_tensor, size=target_size, mode='bilinear', align_corners=False
                )
                
                # Save reduced file
                torch.save(resized_tensor, save_file_path)
                print(f"Saved: {save_file_path}")

# Example usage
train_dataset_path = 'H:/Datasets/int_split/Training/'
val_dataset_path = 'H:/Datasets/int_split/Validation/'
test_dataset_path = 'H:/Datasets/int_split/Testing/'

train_save_path = 'H:/Datasets/reduced/Training/'
val_save_path = 'H:/Datasets/reduced/Validation/'
test_save_path = 'H:/Datasets/reduced/Testing/'

process_and_save(train_dataset_path, train_save_path)
process_and_save(val_dataset_path, val_save_path)
process_and_save(test_dataset_path, test_save_path)