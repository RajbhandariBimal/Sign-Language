"""
ASL Alphabet Real-time Detection Script

This script uses a trained CNN model to recognize ASL alphabet letters in real-time
using a webcam feed.

Installation Requirements:
pip install opencv-python tensorflow keras numpy

Prerequisites:
1. Run train_model.py first to train and save the model
2. Ensure asl_model.h5 and class_names.npy files exist in the same directory
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

class ASLDetector:
    def __init__(self, model_path='asl_model.h5', labels_path='class_names.npy'):
        """
        Initialize the ASL detector
        
        Args:
            model_path (str): Path to the trained model file
            labels_path (str): Path to the class names file
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_names = None
        self.img_size = (200, 200)
        self.confidence_threshold = 0.9  # 90% confidence threshold
        
        # Load model and class names
        self.load_model_and_labels()
        
        # Initialize webcam
        self.cap = None
        
    def load_model_and_labels(self):
        """
        Load the trained model and class labels
        """
        try:
            print("Loading trained ASL model...")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file '{self.model_path}' not found. "
                                      f"Please run train_model.py first.")
            
            self.model = keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
            
            print("Loading class names...")
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"Labels file '{self.labels_path}' not found. "
                                      f"Please run train_model.py first.")
            
            self.class_names = np.load(self.labels_path)
            print(f"Loaded {len(self.class_names)} classes: {list(self.class_names)}")
            
        except Exception as e:
            print(f"Error loading model or labels: {e}")
            raise
    
    def preprocess_frame(self, frame, roi_coords=None):
        """
        Preprocess a frame for ASL recognition
        
        Args:
            frame: Input frame from webcam
            roi_coords: Region of interest coordinates (x, y, w, h)
            
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        """
        # If ROI coordinates are provided, crop the frame
        if roi_coords:
            x, y, w, h = roi_coords
            frame = frame[y:y+h, x:x+w]
        
        # Resize to model input size
        frame_resized = cv2.resize(frame, self.img_size)
        
        # Convert BGR to RGB (if needed)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict_asl(self, frame, roi_coords=None):
        """
        Predict ASL letter from a frame
        
        Args:
            frame: Input frame from webcam
            roi_coords: Region of interest coordinates (x, y, w, h)
            
        Returns:
            tuple: (predicted_letter, confidence)
        """
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame, roi_coords)
        
        # Make prediction
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Return letter if confidence is above threshold, otherwise "Unknown"
        if confidence >= self.confidence_threshold:
            predicted_letter = self.class_names[predicted_class_idx]
        else:
            predicted_letter = "Unknown"
        
        return predicted_letter, confidence
    
    def draw_roi_rectangle(self, frame, roi_coords):
        """
        Draw region of interest rectangle on frame
        
        Args:
            frame: Input frame
            roi_coords: ROI coordinates (x, y, w, h)
            
        Returns:
            frame: Frame with ROI rectangle drawn
        """
        x, y, w, h = roi_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand here", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
    
    def draw_prediction(self, frame, prediction, confidence):
        """
        Draw prediction text on frame
        
        Args:
            frame: Input frame
            prediction: Predicted letter
            confidence: Prediction confidence
            
        Returns:
            frame: Frame with prediction text
        """
        # Choose color based on confidence
        if prediction == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            text = f"Letter: {prediction}"
        else:
            color = (0, 255, 0)  # Green for confident prediction
            text = f"Letter: {prediction} ({confidence:.2f})"
        
        # Draw background rectangle for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 30), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame
    
    def draw_instructions(self, frame):
        """
        Draw instructions on frame
        
        Args:
            frame: Input frame
            
        Returns:
            frame: Frame with instructions
        """
        instructions = [
            "ASL Alphabet Recognition",
            "Place your hand in the green rectangle",
            "Press 'q' to quit, 'c' to toggle confidence threshold"
        ]
        
        y_offset = frame.shape[0] - 80
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_detection(self):
        """
        Run real-time ASL detection using webcam
        """
        print("Starting ASL detection...")
        print("Controls:")
        print("- Place your hand in the green rectangle")
        print("- Press 'q' to quit")
        print("- Press 'c' to toggle confidence threshold (0.9 <-> 0.5)")
        print("- Press 's' to save current frame")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Define ROI (Region of Interest) coordinates
        # Center rectangle where user should place their hand
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        roi_size = 300
        roi_x = (frame_width - roi_size) // 2
        roi_y = (frame_height - roi_size) // 2
        roi_coords = (roi_x, roi_y, roi_size, roi_size)
        
        frame_count = 0
        fps_counter = time.time()
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw ROI rectangle
            frame = self.draw_roi_rectangle(frame, roi_coords)
            
            # Make prediction every few frames for better performance
            if frame_count % 3 == 0:  # Predict every 3rd frame
                prediction, confidence = self.predict_asl(frame, roi_coords)
            
            # Draw prediction on frame
            frame = self.draw_prediction(frame, prediction, confidence)
            
            # Draw instructions
            frame = self.draw_instructions(frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_counter)
                fps_counter = current_time
                
            if frame_count > 30:
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('ASL Alphabet Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Toggle confidence threshold
                if self.confidence_threshold == 0.9:
                    self.confidence_threshold = 0.5
                    print("Confidence threshold changed to 0.5")
                else:
                    self.confidence_threshold = 0.9
                    print("Confidence threshold changed to 0.9")
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                filename = f"asl_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("ASL detection stopped.")

def main():
    """
    Main function to run ASL detection
    """
    print("ASL Alphabet Real-time Detection")
    print("=" * 40)
    
    try:
        # Initialize detector
        detector = ASLDetector()
        
        # Run detection
        detector.run_detection()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'train_model.py' first to train and save the model.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
