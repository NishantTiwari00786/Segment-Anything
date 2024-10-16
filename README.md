
# Nationwide Farmland Dataset Exploration

## Project Overview
This project focuses on geospatial data exploration, leveraging machine learning techniques to analyze and classify agricultural land across the nation. The main goal of the project was to develop a **CNN-based classifier** and utilize META's **Segment Anything Model (SAM)** for segmentation and classification of agricultural lands vs. metropolitan areas from satellite imagery.

### Key Objectives:
1. Build a CNN model to classify satellite images as farmland or metropolitan areas.
2. Customize and integrate the SAM model to assist in segmenting geospatial data for analysis.
3. Develop a pipeline that efficiently processes large geospatial datasets for nationwide agricultural analysis.
4. Modify and test the models on both small and large datasets to ensure scalability and accuracy.

## Project Components
### 1. Data Preparation:
- **Dataset:** Nationwide geospatial data including satellite imagery of both agricultural land (positive examples) and metropolitan areas (negative examples).
- **Preprocessing:** Conversion of **Cropland Data Layer (CDL)** into RGB images for SAM processing.
- **Tools:** Python, GDAL, and various image-processing libraries for geospatial data handling.

### 2. CNN Model (Farmland Classifier):
- **Model Architecture:** A convolutional neural network (CNN) trained to classify satellite images as farmland or non-farmland.
- **Training Data:** Positive examples (agricultural land) and negative examples (metropolitan areas), with data augmentation to expand the dataset.
- **Training Strategy:** Initially tested on smaller datasets, then scaled to larger, nationwide datasets to ensure generalization.
  
### 3. SAM (Segment Anything Model) Integration:
- **Model Used:** ViT-H model from META's SAM framework.
- **Purpose:** To generate segmentation masks for each image and assist the classifier by focusing on specific regions of interest within satellite images.
- **Customization:** Adjusted SAM to process custom geospatial data and integrate with the CNN classifier.

### 4. Polygon Extraction Algorithm:
- Extracts polygons from shapefiles (CA_farmland) and clips them into 256x256 tiles.
- Processes polygons and associated GeoTIFF files, then runs classification on each tile.

### 5. Pipeline:
- **Input:** Shapefile of farmland and a GeoTIFF image.
- **Processing:** Polygon extraction, tiling, SAM segmentation, and CNN classification.
- **Output:** Classified images, labeled as farmland or metropolitan areas.

## Directory Structure
```plaintext
├── CNN_Positive/           # Directory for positive (farmland) training images
├── CNN_Negative/           # Directory for negative (metropolitan) training images
├── Positive_valid/         # Directory for positive validation images
├── Negative_valid/         # Directory for negative validation images
├── US_Basemap/             # Contains GeoTIFF files for analysis
├── CA_farmland.shp         # Shapefile containing CA farmland polygons
├── SAM_weights/            # Pre-trained weights for the SAM model
├── Segment_anything/       # SAM model code directory
├── CDL_dataset/            # Contains Cropland Data Layer (CDL) files
└── output/                 # Directory for storing output images and results
```

## Installation & Setup

### Prerequisites:
- Python 3.x
- Required Python packages: 
  - `gdal`
  - `torch`, `torchvision`
  - `segment-anything`
  - `opencv-python`
  - `matplotlib`

### Steps:
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-repo/nationwide-farmland-dataset.git
   ```
2. Navigate to the project directory:
   ```bash
   cd nationwide-farmland-dataset
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the preprocessing script to convert the CDL dataset to RGB:
   ```bash
   python preprocess_cdl.py --input_path /path/to/cdl.tif --output_path /path/to/output
   ```
5. Train the CNN classifier on your dataset:
   ```bash
   python train_farmland_classifier.py
   ```
6. Use SAM for segmentation on your images:
   ```bash
   python run_sam.py --input_dir /path/to/images --output_dir /path/to/output_masks
   ```

## Results
### Model Performance:
- **Training Accuracy:** X%
- **Validation Accuracy:** X%
- **Test Accuracy on Full Dataset:** X%

### Example Output:
[Insert sample images showing classified farmland and metropolitan areas]

## Future Work
- Expand the model to classify more types of land use beyond agricultural and metropolitan areas.
- Fine-tune SAM for better segmentation accuracy on complex geospatial datasets.
- Apply the model to larger, multi-year satellite datasets for trend analysis.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to submit a pull request or open an issue.

## Acknowledgements
- Thanks to the **University of California, Riverside** for providing the resources and support for this research.
- Special thanks to **META** for the **Segment Anything Model**.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
