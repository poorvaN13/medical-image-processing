import cv2
import numpy as np

class FeatureExtractor:

    def extract(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5,5), 0)

        # Canny edges
        edges = cv2.Canny(blurred, 30, 90)

        # --- Extract 6 features ---

        # 1. Contrast score — std deviation of pixel values
        contrast = float(np.std(enhanced))

        # 2. Edge density — % of pixels that are edges
        edge_density = float(np.count_nonzero(edges) / edges.size * 100)

        # 3. Mean intensity — average brightness
        mean_intensity = float(np.mean(enhanced))

        # 4. SNR — signal to noise ratio
        signal = np.mean(gray)
        noise = np.std(gray - enhanced.astype(np.float32))
        snr = float(20 * np.log10(signal / noise)) if noise != 0 else 0.0

        # 5. Texture — measure of local variation (tumor tissue is irregular)
        texture = float(np.std(np.diff(enhanced.astype(np.float32))))

        # 6. Bright region ratio — tumors often have bright regions
        bright_pixels = np.sum(enhanced > 150)
        bright_ratio = float(bright_pixels / enhanced.size * 100)

        return [contrast, edge_density, mean_intensity, snr, texture, bright_ratio]