import cv2
import numpy as np
import os

class TumorDetector:

    def detect(self, image_path):
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE first (same as our enhancement pipeline)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur to remove noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Thresholding — separates bright tumor region from dark background
        # Anything brighter than threshold becomes white, rest becomes black
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours — outlines of white regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours — ignore very small ones (noise)
        # Only keep contours with area > 500 pixels
        # Get image area
        image_area = image.shape[0] * image.shape[1]
# Keep contours that are between 1% and 30% of image size — tumor sized regions
        significant = [c for c in contours if image_area * 0.01 < cv2.contourArea(c) < image_area * 0.30]

        return image, thresh, significant