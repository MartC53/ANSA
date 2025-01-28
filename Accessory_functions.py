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

train_label_counts = get_label_distribution(train_dataset)
val_label_counts = get_label_distribution(val_dataset)
test_label_counts = get_label_distribution(test_dataset)

def plot_label_distribution(label_counts, title):
    labels, counts = zip(*sorted(label_counts.items()))  
    custom_labels = [label_mapping[label] for label in labels] 
    
    plt.bar(custom_labels, counts)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()
