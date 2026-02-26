import cv2
import numpy as np
import os

class Visualizer:

    def draw_detections(self, image, contours, save_path):
        # Make a copy so we don't modify the original
        output = image.copy()

        if len(contours) == 0:
            print("No tumor regions detected.")
        else:
            # Draw each contour in RED on the image
            cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

            # Draw a bounding box around the LARGEST contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add text label
            cv2.putText(output, "Tumor Region", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save the result
        cv2.imwrite(save_path, output)
        return output