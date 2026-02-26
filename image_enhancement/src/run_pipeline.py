import cv2
import os
from pipeline import ImageEnhancementPipeline
from metrics import Metrics

# Paths to all 4 folders
data_folders = {
    "Train_Normal"     : r"../../data/Brain MRI Images/Train/Normal",
    "Train_Tumor"      : r"../../data/Brain MRI Images/Train/Tumor",
    "Validation_Normal": r"../../data/Brain MRI Images/Validation/Normal",
    "Validation_Tumor" : r"../../data/Brain MRI Images/Validation/Tumor"
}

# Initialize pipeline and metrics
pipeline = ImageEnhancementPipeline(output_dir="../../output")
metrics  = Metrics()

# Loop through all 4 folders
for folder_name, folder_path in data_folders.items():
    print(f"\n--- Processing {folder_name} ---")
    
    images = [f for f in os.listdir(folder_path) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in images:
        image_path = os.path.join(folder_path, image_file)
        
        # Pass folder_name so outputs are saved separately
        result = pipeline.run(image_path, folder_name)
        if result is None:
            continue
            
        gray, enhanced, blurred, edges, quantized = result
        
        scores = metrics.compute_all(gray, enhanced, edges)
        print(f"{image_file}: {scores}")

print("\nDone! Check output/ folder for results.")