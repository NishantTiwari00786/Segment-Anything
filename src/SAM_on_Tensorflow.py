import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Add the parent directory to PYTHONPATH to allow relative imports
import sys
sys.path.append('/Users/nishanttiwari/Desktop/segment-anything')

# Import necessary components
from segment_anything import SamPredictor, sam_model_registry

# Load the SAM model with ViT-H weights
checkpoint_path = '/Users/nishanttiwari/Desktop/SAM weights/sam_vit_h_4b8939.pth'  # Path to your checkpoint
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)

# Function to load image tiles
def load_tiles(tile_dir):
    tile_filenames = [os.path.join(tile_dir, f) for f in os.listdir(tile_dir) if f.endswith('.tif')]
    tiles = []
    for filename in tile_filenames:
        with Image.open(filename) as img:
            img = img.convert("RGB")  # Ensure the image is in RGB format
            tile_array = np.array(img)
            print(f"Loaded tile {filename} with shape {tile_array.shape}")
            tiles.append(tile_array)
    return tiles, tile_filenames

# Function to run SAM on a single tile
def run_sam_on_tile(predictor, tile):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ])
    input_tensor = transform(tile).unsqueeze(0)  # Add batch dimension
    print(f"Transformed tensor shape: {input_tensor.shape}")

    # Convert tensor to (H, W, C) format for SAM model
    input_tensor = input_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Convert to (256, 256, 3)
    print(f"Converted tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        predictor.set_image(input_tensor)
        masks, scores, logits = predictor.predict()
    return masks, scores, logits

# Function to visualize and save masks
def visualize_and_save_masks(tile, masks, scores, tile_name, output_dir):
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
    axes[0].imshow(tile)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i + 1].imshow(tile)
        axes[i + 1].imshow(mask, alpha=0.5)
        axes[i + 1].set_title(f"Mask {i} - Score: {score:.4f}")
        axes[i + 1].axis('off')
        output_filename = os.path.join(output_dir, f'sam_output_{tile_name}_mask_{i}.png')
        plt.savefig(output_filename)
        print(f"Saved SAM output for {tile_name} mask {i} with score {score:.4f} at {output_filename}")
    plt.show()

# Paths
tile_dir = '/Users/nishanttiwari/Desktop/SAM_OLD_TILE' # Update this to the path where tiles are stored
output_dir = '/Users/nishanttiwari/Desktop/OLD_SAM_OUTPUTS' # Update this to the path where you want to save SAM results
os.makedirs(output_dir, exist_ok=True)

# Load the tiles
print("Loading tiles...")
tiles, tile_filenames = load_tiles(tile_dir)
print(f"Loaded {len(tiles)} tiles.")

# Load the SAM model
print("Loading SAM model...")
sam_predictor = SamPredictor(sam)
sam.eval()

# Run SAM on each tile
print("Running SAM on tiles...")
for tile, filename in zip(tiles, tile_filenames):
    tile_name = os.path.basename(filename)
    print(f"Processing {tile_name}...")
    try:
        masks, scores, logits = run_sam_on_tile(sam_predictor, tile)
        
        # Visualize and save the masks with scores
        visualize_and_save_masks(tile, masks, scores, tile_name, output_dir)
        
        # Combine masks using a union (logical OR operation)
        combined_mask = np.any(masks, axis=0)

        # Apply the combined mask to the original image
        masked_image = tile * combined_mask[..., np.newaxis]  # Apply mask to each channel of the image

        # Convert the masked image to a format suitable for saving
        masked_image_pil = Image.fromarray((masked_image * 255).astype(np.uint8))

        # Save the masked image as JPEG
        combined_output_path = os.path.join(output_dir, f'{tile_name}_combined_masked.jpg')
        masked_image_pil.save(combined_output_path, 'JPEG')

        print(f"Combined masked image saved at {combined_output_path}")
    
    except ValueError as e:
        print(f"Skipping {tile_name} due to error: {e}")

print("SAM processing complete!")
