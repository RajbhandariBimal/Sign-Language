# ASL Alphabet Recognition System

A complete Python desktop application that recognizes all 26 letters of the ASL (American Sign Language) alphabet in real-time using a webcam and deep learning.

## Features

- **Real-time Recognition**: Recognizes ASL alphabet letters (A-Z) using webcam feed
- **High Accuracy**: Uses a trained Convolutional Neural Network (CNN) with TensorFlow/Keras
- **Confidence Threshold**: Displays "Unknown" for gestures with <90% confidence
- **Cross-platform**: Works on Windows, Mac, and Linux
- **Offline Operation**: Runs completely offline after training

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam for real-time detection

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install opencv-python tensorflow keras numpy scikit-learn matplotlib pillow
```

### Optional: Kaggle API (for automatic dataset download)

```bash
pip install kaggle
```

## Quick Start

### 1. Download the Dataset

**Option A: Automatic Download (Recommended)**

```bash
python setup_dataset.py
```

**Option B: Manual Download**

1. Visit [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)
2. Download and extract the dataset
3. Rename the folder to `asl_alphabet_train`
4. Ensure the folder contains subfolders A, B, C, ..., Z

### 2. Train the Model

```bash
python train_model.py
```

This will:

- Load and preprocess the ASL alphabet dataset
- Train a CNN model for ~30 epochs
- Save the trained model as `asl_model.h5`
- Save class names as `class_names.npy`
- Display training progress and accuracy

### 3. Run Real-time Detection

```bash
python detect_asl.py
```

## Usage

### Training (`train_model.py`)

The training script will:

1. Load images from the `asl_alphabet_train` folder
2. Preprocess images (resize to 200×200, normalize)
3. Train a CNN with data augmentation
4. Save the trained model and class labels

**Training Parameters:**

- Image size: 200×200 pixels
- Epochs: 30 (with early stopping)
- Batch size: 32
- Data augmentation: rotation, shifts, zoom

### Real-time Detection (`detect_asl.py`)

The detection script provides:

- Live webcam feed with ASL recognition
- Green rectangle showing where to place your hand
- Real-time prediction display with confidence score
- FPS counter

**Controls:**

- `q`: Quit the application
- `c`: Toggle confidence threshold (0.9 ↔ 0.5)
- `s`: Save current frame as image

### Dataset Setup (`setup_dataset.py`)

Utility script to automatically download and organize the ASL dataset from Kaggle.

**Kaggle API Setup:**

1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → API → Create New API Token
3. Download `kaggle.json`
4. Place in appropriate location:
   - **Windows**: `C:\Users\{username}\.kaggle\kaggle.json`
   - **Mac/Linux**: `~/.kaggle/kaggle.json`

## Project Structure

```
Sign/
├── train_model.py          # Model training script
├── detect_asl.py          # Real-time detection script
├── setup_dataset.py       # Dataset download utility
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── asl_alphabet_train/   # Dataset folder (after download)
│   ├── A/               # Images for letter A
│   ├── B/               # Images for letter B
│   └── ...              # Images for letters C-Z
├── asl_model.h5          # Trained model (after training)
├── class_names.npy       # Class labels (after training)
└── training_history.png  # Training plots (after training)
```

## Model Architecture

The CNN model consists of:

- 4 Convolutional blocks (32, 64, 128, 256 filters)
- Batch normalization and dropout for regularization
- Max pooling for dimensionality reduction
- Dense layers with 512 and 256 neurons
- Softmax output for 26 classes (A-Z)

**Model Performance:**

- Expected validation accuracy: >95%
- Real-time inference: ~30 FPS
- Model size: ~50MB

## Technical Details

### Image Preprocessing

- Resize to 200×200 pixels
- Normalize pixel values to [0, 1]
- Convert BGR to RGB color space

### Data Augmentation

- Rotation: ±10 degrees
- Width/height shift: ±10%
- Zoom: ±10%
- No horizontal flip (ASL letters are orientation-specific)

### Real-time Processing

- Center crop region of interest (300×300 pixels)
- Process every 3rd frame for performance
- Mirror webcam feed for natural interaction

## Troubleshooting

### Common Issues

**1. "Dataset directory not found"**

- Run `python setup_dataset.py` to download the dataset
- Or manually download and extract to `asl_alphabet_train/`

**2. "Model file not found"**

- Run `python train_model.py` to train and save the model

**3. "Could not open webcam"**

- Check if webcam is connected and not used by other applications
- Try changing webcam index in `detect_asl.py` (line: `cv2.VideoCapture(0)`)

**4. Low accuracy/poor recognition**

- Ensure good lighting conditions
- Place hand clearly within the green rectangle
- Try adjusting confidence threshold with 'c' key

**5. Slow performance**

- Reduce frame processing frequency in `detect_asl.py`
- Close other applications using GPU/CPU resources

### Performance Optimization

**For slower computers:**

- Reduce image size in training (change `img_size` parameter)
- Process fewer frames per second in detection
- Use smaller model architecture

**For better accuracy:**

- Increase training epochs
- Add more data augmentation
- Use transfer learning with pre-trained models

## Hardware Requirements

### Minimum

- CPU: Dual-core processor
- RAM: 4GB
- Storage: 2GB free space
- Webcam: Any USB webcam

### Recommended

- CPU: Quad-core processor or better
- RAM: 8GB or more
- GPU: NVIDIA GPU with CUDA support (optional)
- Storage: 5GB free space
- Webcam: HD webcam for better recognition

## License

This project is for educational purposes. The ASL Alphabet Dataset is provided by Kaggle user grassknoted.

## Contributing

Feel free to improve this project by:

- Adding more ASL gestures (words, phrases)
- Improving model accuracy
- Optimizing real-time performance
- Adding new features (voice output, gesture sequences)

## Acknowledgments

- ASL Alphabet Dataset by [grassknoted](https://www.kaggle.com/grassknoted/asl-alphabet)
- TensorFlow and OpenCV communities
- ASL community for promoting sign language accessibility
