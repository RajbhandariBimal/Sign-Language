"""
Hand Detection and Processing Utilities for ASL Recognition

This module provides utility functions for hand detection, preprocessing,
and region of interest extraction for ASL alphabet recognition.

Optional: Install MediaPipe for enhanced hand detection
pip install mediapipe
"""

import cv2
import numpy as np

# Try to import MediaPipe, fall back to basic detection if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Using basic hand detection.")

class HandDetector:
    """
    Hand detection class with optional MediaPipe support for better hand region extraction
    """
    
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        """
        Initialize hand detector
        
        Args:
            detection_confidence (float): Minimum confidence for hand detection
            tracking_confidence (float): Minimum confidence for hand tracking
        """
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,  # Only detect one hand
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
            print("Using MediaPipe for hand detection")
        else:
            print("Using basic ROI-based hand detection")
        
    def detect_hand(self, frame):
        """
        Detect hand in the frame and return bounding box
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (hand_detected, bbox, landmarks)
                - hand_detected: Boolean indicating if hand was found
                - bbox: Bounding box coordinates (x, y, w, h) or None
                - landmarks: Hand landmarks or None (only with MediaPipe)
        """
        if self.use_mediapipe:
            return self._detect_hand_mediapipe(frame)
        else:
            return self._detect_hand_basic(frame)
    
    def _detect_hand_mediapipe(self, frame):
        """MediaPipe-based hand detection"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calculate bounding box
            h, w, _ = frame.shape
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            return True, bbox, hand_landmarks
        
        return False, None, None
    
    def _detect_hand_basic(self, frame):
        """Basic motion-based hand detection (fallback)"""
        # This is a simple implementation that assumes hand is in center
        # In a real scenario, you might use background subtraction or other techniques
        h, w = frame.shape[:2]
        
        # Define a center region where we expect the hand
        center_size = min(h, w) // 2
        x = (w - center_size) // 2
        y = (h - center_size) // 2
        
        # Simple check: if there's significant variation in the center region
        center_region = frame[y:y+center_size, x:x+center_size]
        
        if center_region.size > 0:
            # Calculate standard deviation as a measure of content
            gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
            std_dev = np.std(gray)
            
            # If there's enough variation, assume hand is present
            if std_dev > 20:  # Threshold for hand presence
                bbox = (x, y, center_size, center_size)
                return True, bbox, None
        
        return False, None, None
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on the frame (only works with MediaPipe)
        
        Args:
            frame: Input frame
            landmarks: Hand landmarks from MediaPipe
            
        Returns:
            frame: Frame with landmarks drawn
        """
        if self.use_mediapipe and landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS
            )
        return frame
    
    def extract_hand_region(self, frame, bbox=None, target_size=(200, 200)):
        """
        Extract hand region from frame
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h) or None for center crop
            target_size: Target size for the extracted region
            
        Returns:
            numpy.ndarray: Extracted and resized hand region
        """
        if bbox is not None:
            x, y, w, h = bbox
            hand_region = frame[y:y+h, x:x+w]
        else:
            # Center crop if no bbox provided
            h, w = frame.shape[:2]
            size = min(h, w)
            start_x = (w - size) // 2
            start_y = (h - size) // 2
            hand_region = frame[start_y:start_y+size, start_x:start_x+size]
        
        # Resize to target size
        if hand_region.size > 0:
            hand_region = cv2.resize(hand_region, target_size)
        
        return hand_region

def preprocess_for_model(image):
    """
    Preprocess image for ASL model inference
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def create_roi_overlay(frame, roi_coords, color=(0, 255, 0), thickness=2):
    """
    Create region of interest overlay on frame
    
    Args:
        frame: Input frame
        roi_coords: ROI coordinates (x, y, w, h)
        color: Rectangle color (B, G, R)
        thickness: Rectangle thickness
        
    Returns:
        frame: Frame with ROI overlay
    """
    x, y, w, h = roi_coords
    
    # Draw main rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw corner markers for better visibility
    corner_length = 20
    corner_thickness = thickness + 1
    
    # Top-left corner
    cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thickness)
    
    # Top-right corner
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
    
    # Bottom-left corner
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
    
    # Bottom-right corner
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)
    
    return frame

def calculate_center_roi(frame_width, frame_height, roi_size=300):
    """
    Calculate centered ROI coordinates
    
    Args:
        frame_width: Frame width
        frame_height: Frame height
        roi_size: Size of the ROI square
        
    Returns:
        tuple: ROI coordinates (x, y, w, h)
    """
    x = (frame_width - roi_size) // 2
    y = (frame_height - roi_size) // 2
    return (x, y, roi_size, roi_size)

def apply_image_enhancements(image):
    """
    Apply image enhancements for better recognition
    
    Args:
        image: Input image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Convert to LAB color space for better contrast adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

if __name__ == "__main__":
    # Test hand detection
    print("Testing hand detection...")
    
    # Initialize hand detector
    detector = HandDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    
    print("Hand detection test running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hand
        hand_detected, bbox, landmarks = detector.detect_hand(frame)
        
        if hand_detected:
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Hand Detected", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw landmarks (if available)
            frame = detector.draw_landmarks(frame, landmarks)
            
            # Extract hand region
            hand_region = detector.extract_hand_region(frame, bbox)
            
            # Show hand region in a separate window
            if hand_region.size > 0:
                cv2.imshow('Hand Region', hand_region)
        else:
            cv2.putText(frame, "No Hand Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw center ROI
        roi_coords = calculate_center_roi(frame.shape[1], frame.shape[0])
        frame = create_roi_overlay(frame, roi_coords)
        
        cv2.imshow('Hand Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Hand detection test completed.")
