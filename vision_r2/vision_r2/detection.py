"""
YOLO object detection utilities
"""
from ultralytics import YOLO
from vision_r2.config import MODEL_PATH, CONF_THRES, COLORS, MIN_VALID_DEPTH


class ObjectDetector:
    """
    Wrapper for YOLO object detection
    """
    
    def __init__(self, model_path=MODEL_PATH, conf_threshold=CONF_THRES):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully")
    
    def detect(self, image):
        """
        Run detection on an image
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            YOLO results object
        """
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        return results[0]
    
    def parse_detections(self, results, depth_frame, depth_processor):
        """
        Parse YOLO results into structured detection data
        
        Args:
            results: YOLO results object
            depth_frame: RealSense depth frame
            depth_processor: Depth processing utility
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            
            # Get depth
            depth_m = depth_processor.get_depth_median(depth_frame, cx, cy)
            
            # Validate depth
            if depth_m is not None and depth_m < MIN_VALID_DEPTH:
                depth_m = None
            
            # Get color for this class
            color = COLORS.get(class_name, COLORS["default"])
            
            detection = {
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "class_name": class_name,
                "class_id": cls_id,
                "confidence": conf,
                "depth": depth_m,
                "color": color,
            }
            
            detections.append(detection)
        
        return detections
    
    def find_best_hand(self, detections):
        """
        Find the best hand detection based on depth or confidence
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Best hand detection dict, or None if no hand found
        """
        hand_candidates = [d for d in detections if d["class_name"] == "hand"]
        
        if not hand_candidates:
            return None
        
        # Prioritize detections with valid depth, pick closest one
        valid_depth = [c for c in hand_candidates if c["depth"] is not None]
        if valid_depth:
            return min(valid_depth, key=lambda c: c["depth"])
        
        # Otherwise, pick highest confidence
        return max(hand_candidates, key=lambda c: c["confidence"])