# Object Detection and Retrieval System

This repository contains a Flask-based web application designed for detecting objects in images and retrieving visually similar images from a dataset using advanced deep learning and computer vision techniques.

## Features

- **Object Detection**: Utilizes YOLOv3 and Detectron2 for detecting objects in uploaded or URL-based images.
- **Image Embedding**: Extracts embeddings using NASNetLarge and other deep learning models.
- **Content-Based Image Retrieval (CBIR)**: Retrieves visually similar images from the dataset using embeddings and re-ranking methods.
- **Re-Ranking**: Refines retrieval results based on dominant color and texture similarity.
- **Elasticsearch Integration**: Manages image indexing and metadata storage.

## Technologies Used

- **Framework**: Flask
- **Deep Learning**: Keras (NASNetLarge, InceptionResNetV2,...)
- **Search Engine**: Elasticsearch
- **Object Detection**: YOLOv3, Detectron2
- **Reranking**: Color and texture-based methods
- **Additional Libraries**: NumPy, OpenCV, skimage

## Prerequisites

- Python 3.8 or higher
- Elasticsearch installed and running
- Required Python packages (install via `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/object-detection-retrieval.git
   cd object-detection-retrieval
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the application:
   - Update the `cfg/config.cfg` file with your settings (proxy, directories, etc.).
   - Ensure Elasticsearch is configured and accessible.

4. Start the Flask application:
   ```bash
   python app.py
   ```

## Usage

### 1. Uploading Images
- Access the upload page via `http://localhost:5002/`.
- Upload an image or provide a URL for analysis.

### 2. Object Detection
- Detected objects and bounding boxes are displayed.

### 3. Image Retrieval
- Retrieves visually similar images based on embeddings and reranking methods (color and texture).

### 4. Displaying Results
- View retrieved images and their details on the web interface.

## API Endpoints

- **`/receive_image`**: Accepts a URL-based image for detection and retrieval.
- **`/visual_search`**: Retrieves similar images based on uploaded or URL images.
- **`/display/<filename>`**: Displays a specific image.
- **`/displays/<filename>`**: Displays images from a category.

## Configuration

Key settings in `cfg/config.cfg`:
- `proxy`: Proxy settings for external requests.
- `request_dir`: Directory for temporary request storage.
- `dataset_folder`: Path to the dataset directory.
- `upload_folder`: Directory for uploaded files.
