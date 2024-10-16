!pip install geopandas rasterio shapely
import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

# Define paths
shapefile_path = '/Users/nishanttiwari/Desktop/CA_farmland(file contents no code)'
geotiff_directory = '/Users/nishanttiwari/Desktop/US_BASEMAP'
output_directory = '/Users/nishanttiwari/Desktop/Extraction_CA'

# Load the shapefile
shapefile = gpd.read_file(shapefile_path)
if shapefile.empty:
    print("Shapefile is empty or not loaded properly.")
else:
    print("Shapefile loaded successfully. CRS:", shapefile.crs)

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print("Created output directory:", output_directory)
else:
    print("Output directory already exists:", output_directory)

# Process each GeoTIFF file in the directory
for file_name in os.listdir(geotiff_directory):
    if file_name.endswith('.tif'):
        raster_path = os.path.join(geotiff_directory, file_name)
        with rasterio.open(raster_path) as src:
            print(f"Processing {file_name} with CRS: {src.crs}")

            # Check and align CRS
            if shapefile.crs != src.crs:
                print("CRS mismatch found. Reprojecting shapefile...")
                transformed_shapefile = shapefile.to_crs(src.crs)
            else:
                transformed_shapefile = shapefile

            # Convert raster bounds to a geometry
            raster_bounds = box(*src.bounds)

            # Filter to only include geometries that intersect with raster bounds
            intersecting_geometries = transformed_shapefile[transformed_shapefile.geometry.intersects(raster_bounds)]
            print(f"Found {len(intersecting_geometries)} intersecting geometries.")

            # Process each intersecting geometry
            for index, geom in intersecting_geometries.iterrows():
                out_image, out_transform = mask(src, [geom['geometry']], crop=True)
                if out_image.size > 0:
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })

                    output_file_path = os.path.join(output_directory, f'output_{geom["OBJECTID"]}_{file_name}')
                    with rasterio.open(output_file_path, 'w', **out_meta) as dst:
                        dst.write(out_image)
                    print(f"Output written to {output_file_path}")
                else:
                    print(f"No data within the bounds for geometry {index}")

print("Processing complete.")
