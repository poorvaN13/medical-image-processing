import cv2
import numpy as np
import os

class ImageEnhancementPipeline:
    
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_clahe(self, gray, clip_limit=2.0, tile_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(gray)

    def apply_gaussian_blur(self, image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_canny(self, blurred, low_thresh=30, high_thresh=90):
        return cv2.Canny(blurred, low_thresh, high_thresh)

    def apply_quantization(self, image, levels=8):
        step = 256 // levels
        return (image // step) * step

    def run(self, image_path, folder_name):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load {image_path}")
            return

        # Get just the filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Create subfolder for this category e.g. output/Train_Normal/
        save_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # Run all 5 stages
        gray      = self.to_grayscale(image)
        enhanced  = self.apply_clahe(gray)
        blurred   = self.apply_gaussian_blur(enhanced)
        edges     = self.apply_canny(blurred)
        quantized = self.apply_quantization(enhanced)

        # Save each stage with the image filename
        cv2.imwrite(f"{save_dir}/{filename}_1_grayscale.png", gray)
        cv2.imwrite(f"{save_dir}/{filename}_2_clahe.png", enhanced)
        cv2.imwrite(f"{save_dir}/{filename}_3_blur.png", blurred)
        cv2.imwrite(f"{save_dir}/{filename}_4_edges.png", edges)
        cv2.imwrite(f"{save_dir}/{filename}_5_quantized.png", quantized)

        print(f"Saved: {filename}")
        return gray, enhanced, blurred, edges, quantized