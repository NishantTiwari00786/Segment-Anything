!pip install opencv-python numpy matplotlib Pillow torchvision
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from PIL import Image  # Import the Image class from PIL to save images

# Add the path to the 'segment-anything' directory to PYTHONPATH
sys.path.append('/Users/nishanttiwari/Desktop/segment-anything')

# Import necessary components from the 'segment-anything' module
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

# Load the SAM model with ViT-H
checkpoint_path = '/Users/nishanttiwari/Desktop/SAM weights/sam_vit_h_4b8939.pth'
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)

# Example 1: Using a prompt to generate masks
predictor = SamPredictor(sam)

# Load your image
image_path =  '/Users/nishanttiwari/Desktop/output_256_1024_328-1343_quad.tif' # Update with your image path
image = cv2.imread(image_path)
predictor.set_image(image)

# Convert point_coords to a NumPy array to avoid AttributeError
input_prompts = {"point_coords": np.array([[100, 100]]), "point_labels": [1]}
masks, _, _ = predictor.predict(**input_prompts)

# Example 2: Generating masks for the entire image automatically
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Manually set the output directory
output_dir = '/Users/nishanttiwari/Desktop/OPEN_CV_SAM_OUTPUT'  # Fixed directory path
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists without asking for confirmation

# Save the masks to files
for i, mask in enumerate(masks):
    mask_image = (mask["segmentation"] * 255).astype(np.uint8)  # Convert mask to uint8
    mask_image_path = os.path.join(output_dir, f'sam_output_mask_{i}.png')
    cv2.imwrite(mask_image_path, mask_image)
    print(f"Mask {i} saved at {mask_image_path}")

# Display the original image with masks using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Iterate through the masks and plot them with confidence levels
for i, mask in enumerate(masks):
   plt.contour(mask["segmentation"], colors='r')
    # Display the confidence score (predicted IoU) on the mask
  #  if "predicted_iou" in mask:
   #     plt.text(mask["bbox"][0], mask["bbox"][1] - 10, 
    #             f"Conf: {mask['predicted_iou']:.2f}",
     #            color='white', fontsize=8, backgroundcolor='black')

plt.show()

print("Script executed successfully!")

# Combine masks using a union (logical OR operation)
combined_mask = np.any([mask['segmentation'] for mask in masks], axis=0)

# Apply the combined mask to the entire image
masked_image = cv2.bitwise_and(image, image, mask=combined_mask.astype(np.uint8))

# Save the masked image
output_image_path = os.path.join(output_dir, 'masked_full_image.jpg')
output_image = Image.fromarray(masked_image)
output_image.save(output_image_path)
print(f"Masked image saved as {output_image_path}")
