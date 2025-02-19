# Plant Disease Recognition System 🌿

A deep learning-based web application that uses Convolutional Neural Networks (CNN) to detect plant diseases from leaf images.

## Overview

The Plant Disease Recognition System is an AI-powered tool that helps farmers and gardeners identify plant diseases quickly and accurately. The system uses a trained TensorFlow model to analyze images of plant leaves and detect 38 different types of plant diseases across various crops.

## Features

- Real-time plant disease detection
- Support for multiple crops including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato
- User-friendly Streamlit web interface
- Fast and accurate predictions
- Comprehensive disease classification system

### CNN Model Structure
- Input Layer: 128x128x3 (RGB images)
- Multiple Convolutional Layers with increasing filters (32 → 512)
- MaxPooling layers for dimensionality reduction
- Dropout layers (0.25, 0.4) for preventing overfitting
- Dense layer with 1500 units
- Output layer with 38 units (softmax activation)

### Training Details
- Optimizer: Adam (learning rate: 0.0001)
- Loss Function: Categorical Crossentropy
- Batch Size: 32
- Image Size: 128x128
- Training Epochs: 10

## Installation Requirements

```bash
pip install streamlit
pip install tensorflow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install pandas
pip install scikit-learn
```

## Dataset

The model is trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data) containing:
- Training set: 70,295 images
- Validation set: 17,572 images
- Test set: 33 images

## Project Structure

```
├── main.py                   # Streamlit web application
├── trained_plant_disease_model.keras  # Trained CNN model
├── training_hist.json        # Training history
├── train/                    # Training dataset
├── valid/                    # Validation dataset
└── test/                     # Test dataset
```

## Usage

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit application:
   ```bash
   streamlit run main.py
   ```

3. Navigate through the application using the sidebar:
   - **Home**: Overview and introduction
   - **About**: Dataset and project information
   - **Disease Recognition**: Upload and analyze plant images

4. To analyze a plant:
   - Go to the "Disease Recognition" page
   - Upload an image using the file uploader
   - Click "Show Image" to verify your upload
   - Click "Predict" to get the disease classification

## Model Evaluation

The model has been evaluated using:
- Training and validation accuracy
- Confusion matrix
- Classification report (Precision, Recall, F1-Score)
- Visual accuracy plots

## Acknowledgments

- Dataset source: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)
- TensorFlow and Keras teams
- Streamlit community
