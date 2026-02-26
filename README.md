# Medical Image Processing
Computer vision projects using Python and OpenCV.

🧠 Medical Image Processing Pipeline
Brain MRI Analysis using Python & OpenCV

Overview
A modular computer vision pipeline for Brain MRI image analysis, covering preprocessing, tumor detection, and classification. Built using Python and OpenCV.

Pipeline Architecture
Image Enhancement:	Grayscale, CLAHE, Gaussian Blur, Canny Edges, Quantization
Tumor Detection:	Otsu Thresholding, Contour Detection, Bounding Box
Classification:	    Feature Extraction + Random Forest — 95% Accuracy

Modules
1. Image Enhancement
Preprocessing pipeline for raw Brain MRI scans.

Grayscale     |  cv2.cvtColor	    |   Remove color noise
CLAHE	      |  cv2.createCLAHE    |   Local contrast enhancement
Gaussian Blur |  cv2.GaussianBlur   |	Remove scanner noise
Canny Edges	  |  cv2.Canny	        |   Detect tissue boundaries
Quantization  |	 Pixel manipulation	|   Structural simplification

Metrics computed for each image: SNR, Edge Density, Contrast Score.

2. Tumor Detection
Contour-based tumor region detection using Otsu thresholding.
•	Automatically finds optimal threshold per image
•	Filters contours by size to remove noise
•	Draws bounding box around detected tumor region

3. Normal vs Tumor Classifier
Machine learning classifier trained on extracted image features.

Features:	   Contrast, Edge Density, Mean Intensity, SNR, Texture, Bright Ratio
Model:	       Random Forest (100 estimators)
Accuracy:	   95% on 300+ Brain MRI images
Tumor Recall:  1.00 — zero missed tumors

Tech Stack
•	Python 3.x
•	OpenCV
•	NumPy
•	Scikit-learn
•	Pandas

Results
•	Processed 300+ Brain MRI images through full pipeline
•	Tumor classifier achieved 95% accuracy
•	Tumor recall of 1.00 — no tumor cases missed
•	All results saved to output/ folder as PNG and CSV

Limitations & Future Work
•	Tumor detection uses classical thresholding — production systems would use U-Net deep learning segmentation
•	Dataset contains 2D slices only — 3D volumetric tracking would require datasets like BraTS
•	Classifier uses handcrafted features — CNN-based feature extraction would improve accuracy further
