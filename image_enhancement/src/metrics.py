import cv2
import numpy as np

class Metrics:

    def signal_to_noise_ratio(self, original, processed):
        # SNR measures how much useful signal vs noise exists
        # Higher SNR = cleaner, better quality image
        signal = np.mean(original)
        noise = np.std(original - processed.astype(np.float32))
        if noise == 0:
            return float('inf')
        return 20 * np.log10(signal / noise)

    def edge_density(self, edges):
        # Edge density = what % of pixels are edges
        # Higher = more detail/boundaries detected
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.count_nonzero(edges)
        return (edge_pixels / total_pixels) * 100

    def contrast_score(self, image):
        # Measures contrast using standard deviation of pixel values
        # Higher = more contrast = better visibility of tumor regions
        return round(float(np.std(image)), 2)

    def compute_all(self, original, enhanced, edges):
        # Runs all 3 metrics and returns them as a dictionary
        return {
            "SNR (dB)"        : round(self.signal_to_noise_ratio(original, enhanced), 2),
            "Edge Density (%)" : round(self.edge_density(edges), 2),
            "Contrast Score"  : self.contrast_score(enhanced)
        }