"""
Enhanced ASL Alphabet Real-time Detection Script

This enhanced version uses the hand detection and backend utilities
for improved performance and accuracy.

Installation Requirements:
pip install opencv-python tensorflow keras numpy mediapipe

Prerequisites:
1. Run train_model.py first to train and save the model
2. Ensure asl_model.h5 and class_names.npy files exist in the same directory
"""

import cv2
import numpy as np
import time
import argparse
from hand import HandDetector, preprocess_for_model, create_roi_overlay, calculate_center_roi
from backend import ASLModelManager, PerformanceMonitor, create_prediction_overlay

class EnhancedASLDetector:
    def __init__(self, model_path='asl_model.h5', labels_path='class_names.npy', 
                 use_mediapipe=True, confidence_threshold=0.9):
        """
        Initialize enhanced ASL detector
        
        Args:
            model_path (str): Path to trained model
            labels_path (str): Path to class names
            use_mediapipe (bool): Whether to use MediaPipe for hand detection
            confidence_threshold (float): Confidence threshold for predictions
        """
        # Initialize components
        self.model_manager = ASLModelManager(model_path, labels_path)
        self.hand_detector = HandDetector()
        self.performance_monitor = PerformanceMonitor()
        
        # Settings
        self.confidence_threshold = confidence_threshold
        self.use_hand_detection = use_mediapipe
        self.img_size = (200, 200)
        
        # State variables
        self.current_prediction = "Unknown"
        self.current_confidence = 0.0
        self.prediction_smoothing = []
        self.smoothing_window = 5
        
        # Load model
        if not self.model_manager.load_model():
            raise RuntimeError("Failed to load ASL model")
        
        print("Enhanced ASL Detector initialized successfully!")
    
    def smooth_predictions(self, prediction, confidence):
        """
        Smooth predictions over multiple frames to reduce jitter
        
        Args:
            prediction: Current prediction
            confidence: Current confidence
            
        Returns:
            tuple: (smoothed_prediction, smoothed_confidence)
        """
        # Add current prediction to history
        self.prediction_smoothing.append((prediction, confidence))
        
        # Keep only recent predictions
        if len(self.prediction_smoothing) > self.smoothing_window:
            self.prediction_smoothing = self.prediction_smoothing[-self.smoothing_window:]
        
        # Count occurrences of each prediction
        prediction_counts = {}
        confidence_sum = {}
        
        for pred, conf in self.prediction_smoothing:
            if pred in prediction_counts:
                prediction_counts[pred] += 1
                confidence_sum[pred] += conf
            else:
                prediction_counts[pred] = 1
                confidence_sum[pred] = conf
        
        # Find most common prediction
        if prediction_counts:
            most_common = max(prediction_counts.items(), key=lambda x: x[1])
            smoothed_prediction = most_common[0]
            smoothed_confidence = confidence_sum[smoothed_prediction] / prediction_counts[smoothed_prediction]
            
            # Only return if it appears in at least 40% of recent frames
            if most_common[1] / len(self.prediction_smoothing) >= 0.4:
                return smoothed_prediction, smoothed_confidence
        
        return "Unknown", 0.0
    
    def process_frame(self, frame):
        """
        Process a single frame for ASL detection
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            tuple: (processed_frame, prediction, confidence)
        """
        frame_start = time.time()
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate ROI coordinates
        roi_coords = calculate_center_roi(frame_width, frame_height, 300)
        
        # Draw ROI overlay
        frame = create_roi_overlay(frame, roi_coords)
        
        # Extract hand region
        if self.use_hand_detection:
            # Use hand detection to get better ROI
            hand_detected, bbox, landmarks = self.hand_detector.detect_hand(frame)
            
            if hand_detected:
                # Draw hand detection results
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                frame = self.hand_detector.draw_landmarks(frame, landmarks)
                
                # Extract hand region
                hand_region = self.hand_detector.extract_hand_region(frame, bbox, self.img_size)
            else:
                # Fall back to ROI if no hand detected
                hand_region = self.hand_detector.extract_hand_region(frame, roi_coords, self.img_size)
        else:
            # Use center ROI
            hand_region = self.hand_detector.extract_hand_region(frame, roi_coords, self.img_size)
        
        # Make prediction if we have a valid hand region
        if hand_region.size > 0:
            # Preprocess for model
            processed_image = preprocess_for_model(hand_region)
            
            # Make prediction
            prediction, confidence, pred_time = self.model_manager.predict_single(
                processed_image, self.confidence_threshold
            )
            
            # Apply prediction smoothing
            smoothed_prediction, smoothed_confidence = self.smooth_predictions(prediction, confidence)
            
            # Update current state
            self.current_prediction = smoothed_prediction
            self.current_confidence = smoothed_confidence
        
        # Add prediction overlay
        frame = create_prediction_overlay(frame, self.current_prediction, self.current_confidence)
        
        # Add performance info
        frame_time = time.time() - frame_start
        self.performance_monitor.log_frame(frame_time)
        
        fps = self.performance_monitor.get_current_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add model performance info
        stats = self.model_manager.get_prediction_stats()
        if stats:
            cv2.putText(frame, f"Avg Pred Time: {stats['avg_prediction_time']*1000:.1f}ms", 
                       (frame_width - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, self.current_prediction, self.current_confidence
    
    def add_instructions(self, frame):
        """
        Add instruction text to frame
        
        Args:
            frame: Input frame
            
        Returns:
            frame: Frame with instructions
        """
        instructions = [
            "Enhanced ASL Alphabet Recognition",
            "Place your hand in the region",
            "Controls: 'q' to quit, 'c' to change confidence, 's' to save frame",
            "'h' to toggle hand detection, 'r' to reset smoothing"
        ]
        
        y_offset = frame.shape[0] - 100
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_detection(self):
        """
        Run enhanced real-time ASL detection
        """
        print("Starting Enhanced ASL Detection...")
        print("Controls:")
        print("- 'q': Quit")
        print("- 'c': Toggle confidence threshold (0.9 <-> 0.5)")
        print("- 's': Save current frame")
        print("- 'h': Toggle hand detection")
        print("- 'r': Reset prediction smoothing")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame, prediction, confidence = self.process_frame(frame)
                
                # Add instructions
                processed_frame = self.add_instructions(processed_frame)
                
                # Display frame
                cv2.imshow('Enhanced ASL Recognition', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Toggle confidence threshold
                    if self.confidence_threshold == 0.9:
                        self.confidence_threshold = 0.5
                        print("Confidence threshold: 0.5")
                    else:
                        self.confidence_threshold = 0.9
                        print("Confidence threshold: 0.9")
                elif key == ord('s'):
                    # Save frame
                    timestamp = int(time.time())
                    filename = f"asl_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved: {filename}")
                elif key == ord('h'):
                    # Toggle hand detection
                    self.use_hand_detection = not self.use_hand_detection
                    print(f"Hand detection: {'ON' if self.use_hand_detection else 'OFF'}")
                elif key == ord('r'):
                    # Reset prediction smoothing
                    self.prediction_smoothing = []
                    print("Prediction smoothing reset")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Save performance log
            stats = self.model_manager.get_prediction_stats()
            self.performance_monitor.save_session_log(stats)
            
            print(f"Session completed. Processed {frame_count} frames.")
            if stats:
                print(f"Average FPS: {self.performance_monitor.get_average_fps():.1f}")
                print(f"Average prediction time: {stats['avg_prediction_time']*1000:.1f}ms")
                print(f"Unknown rate: {stats['unknown_rate']:.1f}%")

def main():
    """
    Main function with command line argument support
    """
    parser = argparse.ArgumentParser(description='Enhanced ASL Alphabet Recognition')
    parser.add_argument('--model', default='asl_model.h5', help='Path to model file')
    parser.add_argument('--labels', default='class_names.npy', help='Path to labels file')
    parser.add_argument('--confidence', type=float, default=0.9, help='Confidence threshold')
    parser.add_argument('--no-mediapipe', action='store_true', help='Disable MediaPipe hand detection')
    
    args = parser.parse_args()
    
    try:
        detector = EnhancedASLDetector(
            model_path=args.model,
            labels_path=args.labels,
            use_mediapipe=not args.no_mediapipe,
            confidence_threshold=args.confidence
        )
        
        detector.run_detection()
        
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please run 'train_model.py' first to create the model.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
