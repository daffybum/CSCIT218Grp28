import pickle
import numpy as np

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure all samples have the same shape
filtered_data = []
filtered_labels = []

for i, sample in enumerate(data_dict['data']):
    if len(sample) == 21:  # Check if the sample has 21 landmarks
        filtered_data.append(np.array(sample).flatten())  # Flatten into a 1D array
        filtered_labels.append(data_dict['labels'][i])  # Keep the corresponding label

# Convert to NumPy arrays
data = np.array(filtered_data)
labels = np.array(filtered_labels)

print("Final dataset shape:", data.shape)
