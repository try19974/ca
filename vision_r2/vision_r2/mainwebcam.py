#!/usr/bin/env python3

"""
Webcam YOLO Object Detection and Tracking System
Main application entry point - Simplified for webcam without depth
"""
import numpy as np
import cv2
import time
import json
import os
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, String

from vision_r2.config import (
    WIN_NAME, FULLSCREEN, TRACKING_COLOR, HOLDING_COLOR,
    CUBE_MODEL_PATH, CUBE_CONF_THRES
)
from vision_r2.detection import ObjectDetector
from vision_r2.tracking import ObjectTracker
from vision_r2.visualization import (
    draw_detection_box, draw_tracking_indicator, 
    draw_info_panel
)

spear_map = {"hand": 0.0, "spear": 1.0, "fist": 2.0}

SPEAR_MODEL_PATH = "/home/korn/Documents/GitHub/fibox_abu_2026_R2_robot/ros2_ws/src/vision_r2/vision_r2/best.pt"

clicked_points = []
current_image = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            labels = ["1. Top-Left", "2. Top-Right", "3. Bottom-Left", "4. Bottom-Right"]
            idx = len(clicked_points) - 1
            cv2.circle(current_image, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(current_image, labels[idx], (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(WIN_NAME, current_image)


class WebcamCamera:
    """Wrapper for OpenCV webcam capture"""
    def __init__(self, camera_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def get_frame(self):
        """Get a single frame from webcam"""
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def start(self):
        """Start camera (no-op for webcam)"""
        pass
    
    def stop(self):
        """Stop camera"""
        self.cap.release()


class TargetPublisher(Node):
    def __init__(self):
        super().__init__('spear_target_publisher')
        self.target_data_pub = self.create_publisher(Float32MultiArray, 'data', 10)
        self.offset_publisher_ = self.create_publisher(Point, 'offsets', 10)
        self.cube_pub = self.create_publisher(String, '/xo/detections', 10)

    def publish_cube_detections(self, detections):
        msg = String()
        msg.data = json.dumps(detections)
        self.cube_pub.publish(msg)

    def publish_target(self, spear_id, x, y, confidence=None, h_angle=0.0, v_height=0.0):
        """Publish target data (2D coordinates only for webcam)"""
        conf_val = float(confidence) if confidence is not None else 0.0
        # For webcam: use normalized coordinates instead of 3D
        data_list = [float(spear_id), round(float(x), 4), round(float(y), 4), 0.0, round(conf_val, 4)]
        
        msg = Float32MultiArray()
        msg.data = data_list
        self.target_data_pub.publish(msg)

        if h_angle is not None and v_height is not None:
            offset_msg = Point()
            offset_msg.x = round(float(h_angle), 4)
            offset_msg.y = round(float(v_height), 4)
            offset_msg.z = 0.0
            self.offset_publisher_.publish(offset_msg)


class GridCalibrator:
    def __init__(self):
        self.H = None
        self.H_inv = None
        self.warp_size = 600
        self.calibration_done = False

    def load(self, path='grid_calib.json'):
        if not os.path.exists(path):
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        corners = data['corners']
        src_pts = np.float32([
            corners['top_left'], corners['top_right'],
            corners['bottom_left'], corners['bottom_right']
        ])
        ws = data.get('warp_size', self.warp_size)
        self.warp_size = ws
        dst_pts = np.float32([[0, 0], [ws, 0], [0, ws], [ws, ws]])
        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_inv = np.linalg.inv(self.H)
        self.calibration_done = True
        print(f'Grid calibration loaded from {path}')
        return True

    def save(self, pts, path='grid_calib.json'):
        data = {
            "corners": {
                "top_left": pts[0], "top_right": pts[1],
                "bottom_left": pts[2], "bottom_right": pts[3]
            },
            "warp_size": self.warp_size
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Grid calibration saved to {path}')

    def calibrate_interactive(self, img):
        global clicked_points, current_image
        clicked_points = []
        current_image = img.copy()
        print(">>> Click 4 corners: Top-Left, Top-Right, Bottom-Left, Bottom-Right")
        cv2.imshow(WIN_NAME, current_image)
        cv2.setMouseCallback(WIN_NAME, mouse_callback)
        while len(clicked_points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
        cv2.destroyAllWindows()
        cv2.setMouseCallback(WIN_NAME, lambda *args: None)
        pts = clicked_points
        src_pts = np.float32(pts)
        dst_pts = np.float32([[0, 0], [self.warp_size, 0], [0, self.warp_size], [self.warp_size, self.warp_size]])
        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_inv = np.linalg.inv(self.H)
        self.calibration_done = True
        self.save(pts)
        return True

    def draw_grid(self, img):
        if not self.calibration_done:
            return img
        ws = self.warp_size
        poly_pts = np.int32([[0, 0], [ws, 0], [ws, ws], [0, ws]])
        src_corners = cv2.perspectiveTransform(np.float32([poly_pts]), self.H_inv)[0]
        src_corners = np.int32(src_corners)
        cv2.polylines(img, [src_corners], True, (0, 255, 255), 2)
        for i in range(1, 3):
            p1 = cv2.perspectiveTransform(np.float32([[[i * ws/3, 0]]]), self.H_inv)[0][0]
            p2 = cv2.perspectiveTransform(np.float32([[[i * ws/3, ws]]]), self.H_inv)[0][0]
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 1)
            p3 = cv2.perspectiveTransform(np.float32([[[0, i * ws/3]]]), self.H_inv)[0][0]
            p4 = cv2.perspectiveTransform(np.float32([[[ws, i * ws/3]]]), self.H_inv)[0][0]
            cv2.line(img, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 255, 255), 1)
        return img

    def map_to_grid(self, cx, cy):
        if not self.calibration_done:
            return None
        ws = self.warp_size
        pt_warped = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), self.H)[0][0]
        wx, wy = pt_warped[0], pt_warped[1]
        if 0 <= wx <= ws and 0 <= wy <= ws:
            col = int(wx // (ws / 3))
            row = int(wy // (ws / 3))
            col_names = ["left", "middle", "right"]
            row_names = ["top", "middle", "bottom"]
            if 0 <= row < 3 and 0 <= col < 3:
                return {'row': row_names[row], 'col': col_names[col]}
        return None


def setup_window():
    """Initialize OpenCV display window"""
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def process_frame(
    color_img, 
    detector, 
    tracker, 
    model_mode="spear"
):
    """
    Process a single frame: detect objects and update tracking
    
    Args:
        color_img: Color image (numpy array)
        detector: ObjectDetector instance
        tracker: ObjectTracker instance
        model_mode: Detection mode - "spear" or "off"
    
    Returns:
        Processed image with annotations
    """
    img = color_img.copy()
    img_h, img_w = img.shape[:2]
    
    detections = []
    
    # Run detection
    if model_mode == "spear":
        results = detector.detect(color_img)
        # Parse detections without depth frame
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            cls_name = detector.model.names[cls_idx]
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "class_name": cls_name,
                "confidence": conf,
                "color": (0, 255, 0),  # Default green
                "depth": None  # No depth for webcam
            })
    
    # Draw all detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]
        
        # Create label without depth
        label = f"{det['class_name'].upper()} {det['confidence']*100:.0f}%"
        
        # Draw detection
        draw_detection_box(
            img, x1, y1, x2, y2, 
            label, det["color"], det["confidence"]
        )
    
    # Find and track best target
    best_target = None
    target_classes = ["hand", "spear", "fist"]
    
    valid_targets = []
    if target_classes:
        valid_targets = [d for d in detections if d["class_name"] in target_classes]
    else:
        valid_targets = detections

    if len(valid_targets) > 0:
        screen_cx = img_w / 2.0
        screen_cy = img_h / 2.0
        best_target = min(
            valid_targets, 
            key=lambda t: (t["center"][0] - screen_cx)**2 + (t["center"][1] - screen_cy)**2
        )

    if best_target is not None:
        cx, cy = best_target["center"]
        conf = best_target["confidence"]
        
        # Store spear type
        tracker.current_spear_type = best_target["class_name"]
        
        # Normalize coordinates to [-1, 1] for angle/height offsets
        h_angle = (cx - img_w/2.0) / (img_w/2.0)  # -1 to 1
        v_height = (cy - img_h/2.0) / (img_h/2.0)  # -1 to 1
        
        # Update tracker with normalized 2D coordinates
        tracker.update_found(cx/img_w, cy/img_h, None, None, conf, h_angle, v_height)
    else:
        tracker.update_lost()
    
    # Draw tracking indicator
    if tracker.is_tracking():
        state_color = TRACKING_COLOR if tracker.found_this_frame else HOLDING_COLOR
        draw_tracking_indicator(img, int(tracker.cx * img_w), int(tracker.cy * img_h), tracker.found_this_frame, state_color)
    
    return img


def process_cube_frame(color_img, detector, grid):
    img = color_img.copy()
    grid.draw_grid(img)

    results = detector.detect(color_img)
    cube_detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = detector.model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        grid_pos = grid.map_to_grid(cx, cy)
        cell_txt = f"{grid_pos['row']}_{grid_pos['col']}" if grid_pos else "Outside"

        cube_detections.append({
            'class': cls_name,
            'confidence': confidence,
            'grid_position': grid_pos
        })

        color = (255, 0, 0) if "blue" in cls_name.lower() else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(img, f"{cls_name} [{cell_txt}]", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img, cube_detections


def main(args=None):
    """Main application loop"""
    print("Initializing ROS...")
    try:
        rclpy.init(args=args)
        ros_node = TargetPublisher()
        print("✓ ROS initialized")
    except Exception as e:
        print(f"✗ ROS init failed: {e}")
        return
    
    # Initialize components
    print("Initializing camera...")
    try:
        camera = WebcamCamera(camera_id=0)
        print("✓ Camera initialized")
    except Exception as e:
        print(f"✗ Camera init failed: {e}")
        return
    
    print("Loading detector...")
    try:
        detector = ObjectDetector(SPEAR_MODEL_PATH, conf_threshold=0.70)
        print("✓ Detector loaded")
    except Exception as e:
        print(f"✗ Detector load failed: {e}")
        camera.stop()
        return
    
    print("Initializing tracker...")
    try:
        tracker = ObjectTracker("hand")
        print("✓ Tracker initialized")
    except Exception as e:
        print(f"✗ Tracker init failed: {e}")
        camera.stop()
        return
    
    # Initialize grid calibrator for cube detection
    grid = GridCalibrator()
    grid.load()
    
    cube_detector = None
    cube_model_loading = False
    
    # Start camera
    print("Starting camera...")
    camera.start()
    
    # Setup display
    print("Setting up display window...")
    setup_window()
    
    # FPS tracking
    prev_time = 0.0
    model_mode = "spear"
    
    print("\n=== Webcam Detection Controls ===")
    print("Q or ESC: Quit")
    print("N: Spear detection mode")
    print("B: Cube detection mode (grid)")
    print("M: Disable all models (OFF)")
    print("==================================\n")
    
    print("Starting main loop... Press Q or ESC to exit")
    
    try:
        frame_count = 0
        while True:
            # Get frame
            color_img = camera.get_frame()
            
            # Check if frame is valid
            if color_img is None:
                print(f"Frame {frame_count}: Failed to capture frame")
                continue
            
            frame_count += 1
            if frame_count == 1:
                print(f"✓ Captured first frame: {color_img.shape}")
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if prev_time > 0 else 0.0
            prev_time = now
            
            mode_label = model_mode.upper()
            processed_img = color_img.copy()
            
            if model_mode == "spear":
                # Process spear detection
                processed_img = process_frame(
                    color_img, 
                    detector, 
                    tracker, 
                    model_mode
                )
                
                # Publish tracking data
                if tracker.is_tracking():
                    state = tracker.get_state()
                    confidence = state.get('confidence', state.get('conf', 0.0))
                    h_angle = state.get('h_angle', getattr(tracker, 'h_angle', 0.0))
                    v_height = state.get('v_height', getattr(tracker, 'v_height', 0.0))
                    spear_name = getattr(tracker, 'current_spear_type', 'unknown')
                    spear_id = spear_map.get(spear_name, -1.0)
                    
                    cx_norm = tracker.cx
                    cy_norm = tracker.cy
                    
                    print(f"✅ Track: ID={spear_id} ({spear_name}), X={cx_norm:.2f}, Y={cy_norm:.2f}, Conf={confidence:.2f}")
                    
                    ros_node.publish_target(spear_id, cx_norm, cy_norm, confidence, h_angle, v_height)
            
            elif model_mode == "cube":
                # Process cube detection
                if not grid.calibration_done:
                    print("No grid calibration. Click 4 corners to calibrate...")
                    if not grid.calibrate_interactive(color_img):
                        print("Calibration cancelled. Switching to spear mode.")
                        model_mode = "spear"
                        continue
                
                if cube_detector is None and not cube_model_loading:
                    cube_model_loading = True
                    print("Loading cube model...")
                    threading.Thread(target=lambda: load_cube_model(), daemon=True).start()
                
                if cube_detector is not None:
                    processed_img, cube_dets = process_cube_frame(color_img, cube_detector, grid)
                    if cube_dets:
                        ros_node.publish_cube_detections(cube_dets)
                else:
                    grid.draw_grid(processed_img)
            
            elif model_mode == "off":
                pass
            
            rclpy.spin_once(ros_node, timeout_sec=0.001)
            
            # Draw mode label on the image
            cv2.putText(processed_img, f"MODE: {mode_label}", (10, processed_img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw info panel for spear mode
            if model_mode == "spear":
                draw_info_panel(processed_img, fps, tracker.get_state())
            
            # Display
            cv2.imshow(WIN_NAME, processed_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("m") or key == ord("M"):  # Toggle to OFF mode
                model_mode = "off"
                print("Model disabled (OFF mode)")
            elif key == ord("n") or key == ord("N"):  # Switch to spear mode
                model_mode = "spear"
                print("Switched to spear detection mode")
            elif key == ord("b") or key == ord("B"):  # Switch to cube mode
                model_mode = "cube"
                print("Switched to cube detection mode")
                if not grid.calibration_done:
                    print("Need grid calibration first")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete")
        
        if rclpy.ok():
            ros_node.destroy_node()
            rclpy.shutdown()


def load_cube_model():
    global cube_detector
    from vision_r2.detection import ObjectDetector
    try:
        cube_detector = ObjectDetector(CUBE_MODEL_PATH, conf_threshold=CUBE_CONF_THRES)
        print("Cube model loaded")
    except Exception as e:
        print(f"Failed to load cube model: {e}")


if __name__ == "__main__":
    main()
