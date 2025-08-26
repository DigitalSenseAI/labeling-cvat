import os
import numpy as np
import cv2

# ---- Configuration ----
input_folder = "/data/projects/anii-anomalias/datasets/sienz/train/ground_truth"        # Folder with .txt label files
output_folder = "/data/projects/anii-anomalias/datasets/sienz/train/labels_3"        # Folder to save binary masks
img_width = 256                # Set your image width here
img_height = 256               # Set your image height here

os.makedirs(output_folder, exist_ok=True)

# ---- Process all txt files in the input_folder ----
for fname in os.listdir(input_folder):
    if fname.endswith(".txt"):
        txt_path = os.path.join(input_folder, fname)
        # Read label file
        with open(txt_path, "r") as f:
            lines = f.readlines()
        # Create blank mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for line in lines:
            items = line.strip().split()
            coords = [float(x) for x in items[1:]]
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i+1] * img_height)
                points.append([x, y])
            if len(points) > 0:
                polygon = np.array([points], dtype=np.int32)
                cv2.fillPoly(mask, polygon, 255)
        # Save the mask as PNG with the same base name
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_folder, f"{base}.png")
        cv2.imwrite(out_path, mask)

print("Done! Masks saved to", output_folder)

