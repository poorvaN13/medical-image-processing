import os
import csv
from feature_extractor import FeatureExtractor
from classifier import TumorClassifier

# Data folders with labels
# 0 = Normal, 1 = Tumor
data_folders = {
    r"../../data/Brain MRI Images/Train/Normal"     : 0,
    r"../../data/Brain MRI Images/Train/Tumor"      : 1,
    r"../../data/Brain MRI Images/Validation/Normal": 0,
    r"../../data/Brain MRI Images/Validation/Tumor" : 1
}

extractor  = FeatureExtractor()
classifier = TumorClassifier()

# --- Step 1: Extract features from all images ---
print("Extracting features from all images...")
all_features = []
all_labels   = []

for folder_path, label in data_folders.items():
    images = [f for f in os.listdir(folder_path)
              if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in images:
        image_path = os.path.join(folder_path, image_file)
        features = extractor.extract(image_path)

        if features is not None:
            all_features.append(features)
            all_labels.append(label)

print(f"Total images processed: {len(all_features)}")
print(f"Normal: {all_labels.count(0)} | Tumor: {all_labels.count(1)}")

# --- Step 2: Train the classifier ---
print("\nTraining classifier...")
accuracy = classifier.train(all_features, all_labels)

# --- Step 3: Save the model ---
classifier.save()

# --- Step 4: Save results to CSV ---
os.makedirs("../../output", exist_ok=True)
with open("../../output/classification_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Actual", "Predicted", "Confidence"])

    for folder_path, label in data_folders.items():
        images = [fi for fi in os.listdir(folder_path)
                  if fi.endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in images:
            image_path = os.path.join(folder_path, image_file)
            features = extractor.extract(image_path)
            if features is None:
                continue

            predicted, confidence = classifier.predict(features)
            actual = "Tumor" if label == 1 else "Normal"
            writer.writerow([image_file, actual, predicted, f"{confidence:.2f}%"])

print("\nResults saved to output/classification_results.csv")
print("Done!")