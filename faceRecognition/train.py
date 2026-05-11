import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# STEP 1: LOAD DATASET
# ---------------------------
data = []
labels = []
size = (100, 100)

dataset_path = "dataset/"

for label, person in enumerate(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)

        data.append(img.flatten())
        labels.append(label)

data = np.array(data).T
labels = np.array(labels)

print("Data Loaded:", data.shape)


mean_face = np.mean(data, axis=1).reshape(-1, 1)
A = data - mean_face

C = np.dot(A.T, A)

# FIXED LINE
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Extra safety
eigenvectors = np.real(eigenvectors)

# Sort eigenvectors
idx = np.argsort(-eigenvalues)
eigenvectors = eigenvectors[:, idx]

# Try different k values
k_values = [5, 10, 15, 20, 25]
accuracies = []

for k in k_values:
    eigvecs_k = eigenvectors[:, :k]

    eigenfaces = np.dot(A, eigvecs_k)
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    weights = np.dot(eigenfaces.T, A).T

    # Train-test split (60-40)
    X_train, X_test, y_train, y_test = train_test_split(weights, labels, test_size=0.4, random_state=42)

    model = MLPClassifier(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)
    print(f"k={k}, Accuracy={acc}")

# ---------------------------
# STEP 3: FINAL MODEL (use best k)
# ---------------------------
k = 20  # choose best from above

eigvecs_k = eigenvectors[:, :k]
eigenfaces = np.dot(A, eigvecs_k)
eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

weights = np.dot(eigenfaces.T, A).T

model = MLPClassifier(max_iter=500)
model.fit(weights, labels)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, mean_face, eigenfaces, weights), f)

print("Model Saved!")

# ---------------------------
# STEP 4: GRAPH
# ---------------------------
plt.plot(k_values, accuracies)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")

plt.savefig("accuracy_graph.png")   
plt.show()