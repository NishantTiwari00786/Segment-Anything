
# Segment Anything Model (SAM) and Farmland Classifier Project

## Overview
This project leverages the **Segment Anything Model (SAM)** alongside other machine learning tools to handle image segmentation tasks, particularly focused on geospatial data related to farmland. The project integrates SAM with libraries like **OpenCV** and **TensorFlow** to process and classify images, with the overall goal of identifying and analyzing farmland using both image segmentation and classification methods.

## Key Scripts

1. **Extraction_polygon.py**
   - This script is responsible for extracting polygons from geospatial datasets, such as shapefiles. It processes these polygons and prepares them for further analysis or integration with the Segment Anything Model (SAM).

2. **SAM_on_OPENCV.py**
   - This script integrates **SAM (Segment Anything Model)** with **OpenCV** to handle image segmentation. The script uses OpenCV for pre-processing and visualization of images, and then applies SAM to generate segmented regions of interest.

3. **SAM_on_Tensorflow.py**
   - This script uses **SAM** in combination with **TensorFlow** to perform advanced machine learning tasks. After SAM completes the segmentation, TensorFlow is used for further processing, training, or classification, particularly useful for classifying agricultural and metropolitan areas.

4. **Tille_extraction.py**
   - The script for extracting tiles (sub-regions) from large geospatial images or maps. It is designed to split the images into smaller, manageable pieces, allowing for more detailed analysis and processing.

## Collaboration and Additional Repository
The **Farmland Classifier** was developed in parallel with this project through collaboration. It involves the training and testing of a classifier to distinguish between farmland and non-farmland areas. For detailed explanations of the classifier's development, training, and results, please refer to the separate repository that contains the classifier's code and additional documentation.

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/NishantTiwari00786/Full-Stack-Game-Rental-App.git
   ```

2. **Install dependencies**:
   - Ensure that **OpenCV**, **TensorFlow**, and other required libraries are installed.

3. **Run the Scripts**:
   - Follow the comments within each script to run them individually. Make sure the necessary datasets and shapefiles are available for processing.

## Technologies Used
- **Segment Anything Model (SAM)**
- **OpenCV**
- **TensorFlow**
- **Python**
- **Geospatial Data**

## Additional Notes
This project is designed to work with geospatial data, particularly shapefiles and satellite imagery, to classify land usage. The integration of SAM allows for powerful image segmentation, while TensorFlow handles the classification tasks.
