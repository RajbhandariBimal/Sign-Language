"""
ASL Alphabet Recognition Model Training Script

This script trains a CNN model to recognize ASL alphabet letters (A-Z).

Installation Requirements:
pip install opencv-python tensorflow keras numpy scikit-learn matplotlib pillow

Dataset Required:
Download ASL Alphabet Dataset from Kaggle:
https://www.kaggle.com/grassknoted/asl-alphabet

Extract the dataset to a folder named 'asl_alphabet_train' in the same directory as this script.
The folder structure should be:
asl_alphabet_train/
├── A/
├── B/
├── C/
...
└── Z/
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ASLModelTrainer:
    def __init__(self, data_dir='asl_alphabet_train', img_size=(200, 200)):
        """
        Initialize the ASL model trainer
        
        Args:
            data_dir (str): Directory containing ASL alphabet training data
            img_size (tuple): Target image size for training (width, height)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the ASL alphabet dataset
        
        Returns:
            tuple: (X_train, X_val, y_train, y_val) preprocessed training data
        """
        print("Loading and preprocessing ASL alphabet dataset...")
        
        # Check if dataset directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory '{self.data_dir}' not found. "
                                  f"Please download the ASL Alphabet Dataset from Kaggle.")
        
        images = []
        labels = []
        
        # Get all class directories (A-Z)
        class_dirs = [d for d in os.listdir(self.data_dir)
                      if os.path.isdir(os.path.join(self.data_dir, d)) and len(d) == 1 and d.isalpha() and d.isupper()]
        class_dirs.sort()  # Ensure consistent ordering
        
        print(f"Found {len(class_dirs)} classes: {class_dirs}")
        
        # Load images from each class directory
        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(image_files)} images for class '{class_name}'...")
            
            for i, image_file in enumerate(image_files):
                if i % 500 == 0:  # Progress indicator
                    print(f"  Processed {i}/{len(image_files)} images for class '{class_name}'")
                
                image_path = os.path.join(class_path, image_file)
                
                try:
                    # Load and preprocess image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size
                    image = cv2.resize(image, self.img_size)
                    
                    # Normalize pixel values to [0, 1]
                    image = image.astype(np.float32) / 255.0
                    
                    images.append(image)
                    labels.append(class_name)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images total")
        print(f"Image shape: {X.shape}")
        
        # Encode labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"Classes: {self.class_names}")
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        
        return X_train, X_val, y_train, y_val
    
    def create_cnn_model(self, num_classes):
        """
        Create a CNN model for ASL alphabet classification
        
        Args:
            num_classes (int): Number of classes (26 for A-Z)
            
        Returns:
            keras.Model: Compiled CNN model
        """
        print("Creating CNN model...")
        
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=50, batch_size=32):
        """
        Train the CNN model
        
        Args:
            X_train, X_val, y_train, y_val: Training and validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            keras.callbacks.History: Training history
        """
        print(f"Training model for {epochs} epochs...")
        
        # Create the model
        num_classes = len(self.class_names)
        self.model = self.create_cnn_model(num_classes)
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.2, patience=5, min_lr=1e-7, monitor='val_accuracy'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_asl_model.h5', save_best_only=True, monitor='val_accuracy'
            )
        ]
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # ASL letters shouldn't be flipped
            fill_mode='nearest'
        )
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model_and_labels(self, model_path='asl_model.h5', labels_path='class_names.npy'):
        """
        Save the trained model and class labels
        
        Args:
            model_path (str): Path to save the model
            labels_path (str): Path to save the class names
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        print(f"Saving model to {model_path}...")
        self.model.save(model_path)
        
        print(f"Saving class names to {labels_path}...")
        np.save(labels_path, self.class_names)
        
        print("Model and labels saved successfully!")
    
    def plot_training_history(self, history):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
        """
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main():
    """
    Main function to train the ASL alphabet recognition model
    """
    print("ASL Alphabet Recognition Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ASLModelTrainer()
    
    try:
        # Load and preprocess data
        X_train, X_val, y_train, y_val = trainer.load_and_preprocess_data()
        
        # Train the model
        history = trainer.train_model(X_train, X_val, y_train, y_val, epochs=30)
        
        # Save the model and labels
        trainer.save_model_and_labels()
        
        # Plot training history
        trainer.plot_training_history(history)
        
        # Evaluate final model
        val_loss, val_accuracy = trainer.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
        print(f"Final Validation Loss: {val_loss:.4f}")
        
        print("\nModel training completed successfully!")
        print("You can now use 'detect_asl.py' for real-time ASL recognition.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download the ASL Alphabet Dataset from:")
        print("https://www.kaggle.com/grassknoted/asl-alphabet")
        print("Extract it to a folder named 'asl_alphabet_train' in this directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
