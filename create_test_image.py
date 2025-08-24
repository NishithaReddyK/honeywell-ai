import cv2
import numpy as np
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Create a black image with red text
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.putText(img, "Test Alert", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Save as JPG
cv2.imwrite("outputs/sample_alert.jpg", img)

print("✅ Valid sample_alert.jpg created inside outputs/")
