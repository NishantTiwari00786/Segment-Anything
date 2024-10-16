!pip install rasterio
import os
import rasterio
from rasterio.windows import Window

def extract_tiles(geotiff_path, output_dir, tile_size=(256, 256), hop_size=(256, 256)):
    """
    Extracts tiles from a GeoTIFF file and saves them to disk.
    Returns the count of generated tiles.
    """
    tile_count = 0
    base_filename = os.path.splitext(os.path.basename(geotiff_path))[0]  # Get the base filename without extension
    with rasterio.open(geotiff_path) as src:
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": tile_size[0],
            "width": tile_size[1],
            "count": src.count
        })

        for j in range(0, src.height, hop_size[1]):
            for i in range(0, src.width, hop_size[0]):
                window = Window(i, j, tile_size[0], tile_size[1])
                data = src.read(window=window)

                if data.shape[1] == tile_size[0] and data.shape[2] == tile_size[1]:
                    output_file_path = os.path.join(output_dir, f'output_{j}_{i}_{base_filename}.tif')
                    with rasterio.open(output_file_path, 'w', **out_meta) as dest:
                        dest.write(data)
                    tile_count += 1

    return tile_count

def process_directory(directory_path, output_dir, tile_size=(256, 256), hop_size=(256, 256)):
    """
    Processes each GeoTIFF in a directory for tile extraction.
    Prints the total count of all tiles generated from all files.
    """
    total_tiles = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('.tif'):
            geotiff_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            tile_count = extract_tiles(geotiff_path, output_dir, tile_size, hop_size)
            total_tiles += tile_count
            print(f"Finished processing {filename}, produced {tile_count} tiles.")

    print(f"Total tiles generated from all images: {total_tiles}")

# Define your paths and settings
directory_path = '/Users/nishanttiwari/Desktop/US_BASEMAP'
output_dir = '/Users/nishanttiwari/Desktop/US_BASEMAP_TILE'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process the directory
process_directory(directory_path, output_dir)
