import pickle
import cv2
import numpy as np

# ---------------------------
# LOAD MODEL
# ---------------------------
with open("model.pkl", "rb") as f:
    model, mean_face, eigenfaces, weights = pickle.load(f)

# ---------------------------
# LOAD TEST IMAGE
# ---------------------------
img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: test.jpg not found ❌")
    exit()

img = cv2.resize(img, (100, 100))
img = img.flatten().reshape(-1, 1)

# ---------------------------
# PREPROCESS
# ---------------------------
img = img - mean_face

# ---------------------------
# PROJECT INTO PCA SPACE
# ---------------------------
test_weight = np.dot(eigenfaces.T, img)

# ---------------------------
# PREDICT
# ---------------------------
prediction = model.predict(test_weight.T)
print("Predicted Person:", prediction)

# ---------------------------
# IMPOSTER DETECTION
# ---------------------------
distances = np.linalg.norm(weights.T - test_weight, axis=0)

min_dist = np.min(distances)
print("Minimum distance:", min_dist)

threshold = 3000

if min_dist > threshold:
    print("Not an enrolled person ❌")
else:
    print("Recognized person ✅")