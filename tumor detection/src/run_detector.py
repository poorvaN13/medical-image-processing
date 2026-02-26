import cv2
import os
from detector import TumorDetector
from visualizer import Visualizer

data_folders = {
    "Train_Tumor"     : r"../../data/Brain MRI Images/Train/Tumor",
    "Validation_Tumor": r"../../data/Brain MRI Images/Validation/Tumor"
}

detector   = TumorDetector()
visualizer = Visualizer()

for folder_name, folder_path in data_folders.items():
    print(f"\n--- Processing {folder_name} ---")

    save_dir = f"../../output/{folder_name}_detected"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    images = [f for f in os.listdir(folder_path)
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Images found: {len(images)}")

    for image_file in images:
        image_path = os.path.join(folder_path, image_file)

        result = detector.detect(image_path)
        if result is None:
            print(f"Skipped: {image_file}")
            continue

        image, thresh, contours = result

        filename = os.path.splitext(image_file)[0]
        save_path = f"{save_dir}/{filename}_detected.png"
        visualizer.draw_detections(image, contours, save_path)

        print(f"Detected: {image_file} — {len(contours)} regions found")

print("\nDone! Check output/ folder for results.")