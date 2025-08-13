"""
Backend utilities for ASL Alphabet Recognition System

This module provides model loading, prediction utilities, and
performance monitoring functions.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import json
from datetime import datetime
import threading
import queue

class ASLModelManager:
    """
    Manages ASL model loading, prediction, and performance monitoring
    """
    
    def __init__(self, model_path='asl_model.h5', labels_path='class_names.npy'):
        """
        Initialize the ASL model manager
        
        Args:
            model_path (str): Path to the trained model
            labels_path (str): Path to the class names file
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_names = None
        self.img_size = (200, 200)
        
        # Performance monitoring
        self.prediction_times = []
        self.prediction_history = []
        self.confidence_history = []
        
        # Threading for async predictions
        self.prediction_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.prediction_thread = None
        self.stop_thread = False
        
    def load_model(self):
        """
        Load the trained model and class names
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file '{self.model_path}' not found")
            
            print(f"Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            print("✓ Model loaded successfully")
            
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"Labels file '{self.labels_path}' not found")
            
            self.class_names = np.load(self.labels_path)
            print(f"✓ Loaded {len(self.class_names)} class names")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def predict_single(self, image, confidence_threshold=0.9):
        """
        Make a single prediction on an image
        
        Args:
            image: Preprocessed image (batch format)
            confidence_threshold: Minimum confidence for valid prediction
            
        Returns:
            tuple: (predicted_letter, confidence, prediction_time)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        
        prediction_time = time.time() - start_time
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Return letter if confidence is above threshold
        if confidence >= confidence_threshold:
            predicted_letter = self.class_names[predicted_class_idx]
        else:
            predicted_letter = "Unknown"
        
        # Store performance data
        self.prediction_times.append(prediction_time)
        self.prediction_history.append(predicted_letter)
        self.confidence_history.append(confidence)
        
        # Keep only last 100 predictions for memory efficiency
        if len(self.prediction_times) > 100:
            self.prediction_times = self.prediction_times[-100:]
            self.prediction_history = self.prediction_history[-100:]
            self.confidence_history = self.confidence_history[-100:]
        
        return predicted_letter, confidence, prediction_time
    
    def predict_batch(self, images, confidence_threshold=0.9):
        """
        Make predictions on a batch of images
        
        Args:
            images: Batch of preprocessed images
            confidence_threshold: Minimum confidence for valid prediction
            
        Returns:
            list: List of (predicted_letter, confidence) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Make predictions
        predictions = self.model.predict(images, verbose=0)
        
        prediction_time = time.time() - start_time
        
        results = []
        for pred in predictions:
            predicted_class_idx = np.argmax(pred)
            confidence = float(pred[predicted_class_idx])
            
            if confidence >= confidence_threshold:
                predicted_letter = self.class_names[predicted_class_idx]
            else:
                predicted_letter = "Unknown"
            
            results.append((predicted_letter, confidence))
        
        return results, prediction_time
    
    def get_prediction_stats(self):
        """
        Get prediction performance statistics
        
        Returns:
            dict: Performance statistics
        """
        if not self.prediction_times:
            return None
        
        stats = {
            'total_predictions': len(self.prediction_times),
            'avg_prediction_time': np.mean(self.prediction_times),
            'min_prediction_time': np.min(self.prediction_times),
            'max_prediction_time': np.max(self.prediction_times),
            'avg_confidence': np.mean(self.confidence_history),
            'fps_estimate': 1.0 / np.mean(self.prediction_times) if np.mean(self.prediction_times) > 0 else 0,
            'unknown_rate': self.prediction_history.count('Unknown') / len(self.prediction_history) * 100
        }
        
        return stats
    
    def start_async_prediction(self):
        """
        Start asynchronous prediction thread for better performance
        """
        if self.prediction_thread is not None:
            return
        
        self.stop_thread = False
        self.prediction_thread = threading.Thread(target=self._prediction_worker)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
    
    def stop_async_prediction(self):
        """
        Stop the asynchronous prediction thread
        """
        self.stop_thread = True
        if self.prediction_thread is not None:
            self.prediction_thread.join()
            self.prediction_thread = None
    
    def _prediction_worker(self):
        """
        Worker thread for asynchronous predictions
        """
        while not self.stop_thread:
            try:
                # Get prediction request
                request = self.prediction_queue.get(timeout=0.1)
                image, confidence_threshold, request_id = request
                
                # Make prediction
                result = self.predict_single(image, confidence_threshold)
                
                # Send result
                self.result_queue.put((request_id, result))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prediction worker error: {e}")
    
    def async_predict(self, image, confidence_threshold=0.9, request_id=None):
        """
        Queue an asynchronous prediction
        
        Args:
            image: Preprocessed image
            confidence_threshold: Minimum confidence threshold
            request_id: Optional request identifier
        """
        if request_id is None:
            request_id = time.time()
        
        self.prediction_queue.put((image, confidence_threshold, request_id))
        return request_id
    
    def get_async_result(self, timeout=0.01):
        """
        Get result from asynchronous prediction
        
        Args:
            timeout: Maximum time to wait for result
            
        Returns:
            tuple: (request_id, result) or None if no result available
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class PerformanceMonitor:
    """
    Monitor and log system performance
    """
    
    def __init__(self, log_file='asl_performance.json'):
        """
        Initialize performance monitor
        
        Args:
            log_file (str): File to save performance logs
        """
        self.log_file = log_file
        self.session_start = time.time()
        self.frame_times = []
        self.fps_history = []
        self.memory_usage = []
        
    def log_frame(self, frame_time):
        """
        Log frame processing time
        
        Args:
            frame_time (float): Time taken to process frame
        """
        self.frame_times.append(frame_time)
        
        # Calculate FPS
        if len(self.frame_times) >= 10:
            recent_times = self.frame_times[-10:]
            avg_time = np.mean(recent_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.fps_history.append(fps)
        
        # Keep only recent data
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
        if len(self.fps_history) > 50:
            self.fps_history = self.fps_history[-50:]
    
    def get_current_fps(self):
        """
        Get current FPS estimate
        
        Returns:
            float: Current FPS
        """
        if len(self.fps_history) > 0:
            return self.fps_history[-1]
        return 0.0
    
    def get_average_fps(self):
        """
        Get average FPS over session
        
        Returns:
            float: Average FPS
        """
        if len(self.fps_history) > 0:
            return np.mean(self.fps_history)
        return 0.0
    
    def save_session_log(self, additional_data=None):
        """
        Save session performance log
        
        Args:
            additional_data (dict): Additional data to include in log
        """
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'session_duration': time.time() - self.session_start,
            'total_frames': len(self.frame_times),
            'average_fps': self.get_average_fps(),
            'current_fps': self.get_current_fps(),
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
        }
        
        if additional_data:
            session_data.update(additional_data)
        
        # Load existing logs
        logs = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new session
        logs.append(session_data)
        
        # Keep only last 50 sessions
        if len(logs) > 50:
            logs = logs[-50:]
        
        # Save logs
        try:
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error saving performance log: {e}")

class ModelEvaluator:
    """
    Evaluate model performance on test data
    """
    
    def __init__(self, model_manager):
        """
        Initialize model evaluator
        
        Args:
            model_manager: ASLModelManager instance
        """
        self.model_manager = model_manager
    
    def evaluate_on_images(self, image_paths, true_labels=None):
        """
        Evaluate model on a set of images
        
        Args:
            image_paths (list): List of image file paths
            true_labels (list): Optional list of true labels
            
        Returns:
            dict: Evaluation results
        """
        if self.model_manager.model is None:
            raise ValueError("Model not loaded")
        
        predictions = []
        confidences = []
        processing_times = []
        
        for image_path in image_paths:
            try:
                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                image = cv2.resize(image, self.model_manager.img_size)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_normalized = image_rgb.astype(np.float32) / 255.0
                image_batch = np.expand_dims(image_normalized, axis=0)
                
                # Make prediction
                pred, conf, proc_time = self.model_manager.predict_single(image_batch)
                
                predictions.append(pred)
                confidences.append(conf)
                processing_times.append(proc_time)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        results = {
            'total_images': len(predictions),
            'predictions': predictions,
            'confidences': confidences,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'unknown_rate': predictions.count('Unknown') / len(predictions) * 100 if predictions else 0
        }
        
        # Calculate accuracy if true labels provided
        if true_labels and len(true_labels) == len(predictions):
            correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
            results['accuracy'] = correct / len(predictions) * 100
        
        return results

def preprocess_image_for_model(image_path, target_size=(200, 200)):
    """
    Load and preprocess an image for model inference
    
    Args:
        image_path (str): Path to image file
        target_size (tuple): Target image size
        
    Returns:
        numpy.ndarray: Preprocessed image batch
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def create_prediction_overlay(frame, prediction, confidence, position=(15, 35)):
    """
    Create prediction overlay on frame
    
    Args:
        frame: Input frame
        prediction: Predicted letter
        confidence: Prediction confidence
        position: Text position (x, y)
        
    Returns:
        frame: Frame with prediction overlay
    """
    # Choose colors based on prediction
    if prediction == "Unknown":
        color = (0, 0, 255)  # Red
        text = f"Letter: {prediction}"
    else:
        if confidence > 0.95:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.8:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (255, 165, 0)  # Orange for low confidence
        text = f"Letter: {prediction} ({confidence:.2f})"
    
    # Draw background rectangle
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, (position[0] - 5, position[1] - 25), 
                 (position[0] + text_size[0] + 5, position[1] + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame

if __name__ == "__main__":
    # Test model manager
    print("Testing ASL Model Manager...")
    
    manager = ASLModelManager()
    
    if manager.load_model():
        print("✓ Model loaded successfully")
        
        # Test prediction on a dummy image
        dummy_image = np.random.random((1, 200, 200, 3)).astype(np.float32)
        pred, conf, time_taken = manager.predict_single(dummy_image)
        
        print(f"Test prediction: {pred} (confidence: {conf:.3f}, time: {time_taken:.3f}s)")
        
        # Print performance stats
        stats = manager.get_prediction_stats()
        if stats:
            print("Performance stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    else:
        print("✗ Failed to load model")
        print("Please run train_model.py first to create the model.")
