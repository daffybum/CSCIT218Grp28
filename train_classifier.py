import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure all samples have 21 landmarks
filtered_data = []
filtered_labels = []

for i, sample in enumerate(data_dict['data']):
    # Ensure the sample has exactly 21 landmarks
    if isinstance(sample, list) and len(sample) == 21:
        # Ensure each landmark has exactly 2 values (x, y)
        if all(len(landmark) == 2 for landmark in sample):
            filtered_data.append(np.array(sample).flatten())  # Flatten to (42,)
            filtered_labels.append(data_dict['labels'][i])  # Keep the corresponding label

# Convert to NumPy arrays
data = np.array(filtered_data)
labels = np.array(filtered_labels)

print("Final dataset shape:", data.shape)  # Expected: (num_samples, 42)


# Split dataset into training & testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Normalize data (optional, but improves accuracy)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train k-NN classifier
k = 7  # Tune this parameter
model = KNeighborsClassifier(n_neighbors=k, weights='distance')
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_test, y_predict)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save model
with open('knn_model.p', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)
