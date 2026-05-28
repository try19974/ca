#!/usr/bin/env python3

"""
RealSense YOLO Object Detection and Tracking System
Main application entry point
"""
import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Float32MultiArray

from vision_r2.config import (
    WIN_NAME, FULLSCREEN, TRACKING_COLOR, HOLDING_COLOR
)
from vision_r2.camera import RealSenseCamera
from vision_r2.detection import ObjectDetector
from vision_r2.depth_utils import get_depth_median, pixel_to_xyz, create_depth_colormap
from vision_r2.tracking import ObjectTracker
from vision_r2.visualization import (
    draw_detection_box, draw_tracking_indicator, 
    draw_info_panel, create_split_view
)
from vision_r2.geometry_utils import calculate_offsets
spear_map = {"hand": 0.0, "spear": 1.0, "fist": 2.0}

class TargetPublisher(Node):
    def __init__(self):
        super().__init__('spear_target_publisher')
        self.target_data_pub = self.create_publisher(Float32MultiArray, 'data', 10)
        self.offset_publisher_ = self.create_publisher(Point, 'offsets', 10)

    def publish_target(self, spear_id, x, y, z, confidence=None, h_angle=0.0, v_height=0.0):
        conf_val = float(confidence) if confidence is not None else 0.0
        data_list = [float(spear_id),round(float(x), 4),round(float(y), 4),round(float(z), 4),round(conf_val, 4)]
        
        msg = Float32MultiArray()
        msg.data = data_list
        self.target_data_pub.publish(msg)

        if h_angle is not None and v_height is not None:
            offset_msg = Point()
            offset_msg.x = round(float(h_angle), 4)  # ให้ X คือแนวนอน (Pan)
            offset_msg.y = round(float(v_height), 4) # ให้ Y คือแนวตั้ง (Tilt)
            offset_msg.z = 0.0                       # ไม่ได้ใช้ ปล่อยเป็น 0
            self.offset_publisher_.publish(offset_msg)

class DepthProcessor:
    """Helper class to make depth processing accessible"""
    @staticmethod
    def get_depth_median(depth_frame, cx, cy):
        return get_depth_median(depth_frame, cx, cy)


def setup_window():
    """Initialize OpenCV display window"""
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def process_frame(
    color_img, 
    depth_frame, 
    detector, 
    tracker, 
    depth_processor,
    yolo_enabled=True
):
    """
    Process a single frame: detect objects and update tracking
    
    Args:
        color_img: Color image (numpy array)
        depth_frame: RealSense depth frame
        detector: ObjectDetector instance
        tracker: ObjectTracker instance
        depth_processor: DepthProcessor instance
    
    Returns:
        Processed image with annotations
    """
    img = color_img.copy()
    img_h, img_w = img.shape[:2]
    
    detections = []
    # Run detection
    if yolo_enabled:
        results = detector.detect(color_img)
        detections = detector.parse_detections(results, depth_frame, depth_processor)
    

    # Draw all detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]
        
        # Create label
        if det["depth"] is None:
            label = f"{det['class_name'].upper()} {det['confidence']*100:.0f}% | -- m"
        else:
            label = f"{det['class_name'].upper()} {det['confidence']*100:.0f}% | {det['depth']:.2f} m"
        
        # Draw detection
        draw_detection_box(
            img, x1, y1, x2, y2, 
            label, det["color"], det["confidence"]
        )

    
    # Find and track best 
    best_target = None
    
    #listหอก
    target_classes = ["hand", "spear", "fist"] # แก้ตรงนี้ให้ตรงกับ YOLO
    
    valid_targets = []
    if target_classes:
        valid_targets = [d for d in detections if d["class_name"] in target_classes]
    else:
        # ถ้าไม่ได้กำหนด List ถือว่าเอาทุกอย่างที่ตรวจเจอ
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
        depth_m = best_target["depth"]
        conf = best_target["confidence"]
        
        # แอบฝากชื่อชนิดของหอกไว้ในตัว tracker ด้วย (เพื่อเอาไปส่งค่าใน main)
        tracker.current_spear_type = best_target["class_name"]
        # Calculate angle and height offsets
        h_angle = 0.0
        v_height = 0.0
        if depth_m is not None:
            h_angle, v_height = calculate_offsets(cx, cy, img_w, img_h, depth_m)
        
        # Get 3D coordinates
        xyz = None
        if depth_m is not None:
            xyz = pixel_to_xyz(depth_frame, cx, cy, depth_m)

        
        # Update tracker
        tracker.update_found(cx, cy, depth_m, xyz, conf, h_angle, v_height)
    else:
        tracker.update_lost()
    
    # Draw tracking indicator
    if tracker.is_tracking():
        state_color = TRACKING_COLOR if tracker.found_this_frame else HOLDING_COLOR
        draw_tracking_indicator(img, tracker.cx, tracker.cy, tracker.found_this_frame, state_color)
    
    return img


def main(args=None):
    """Main application loop"""
    rclpy.init(args=args)
    ros_node = TargetPublisher()
    # Initialize components
    camera = RealSenseCamera()
    detector = ObjectDetector()
    tracker = ObjectTracker("hand")
    depth_processor = DepthProcessor()
    
    # Start camera
    camera.start()
    
    # Setup display
    setup_window()
    
    # FPS tracking
    prev_time = 0.0

    yolo_enabled = True
    
    print("\n=== Controls ===")
    print("Q or ESC: Quit")
    print("M: Toggle YOLO Model (ON/OFF)") 
    print("===============\n")
    
    
    try:
        while True:
            # Get frames
            # รับค่าทั้งหมดมาก่อน
            frames = camera.get_frames()
        
            # ถ้ากล้อง Error มันจะส่งค่ามาไม่ได้ 2 ตัว ให้สั่งข้ามเฟรมนี้ไปรอก่อน
            if not frames or len(frames) != 2:
                continue
            
            depth_frame, color_frame = frames
        
            # ถ้าภาพมาเป็นค่าว่าง ก็ให้ข้ามไปรอก่อนเช่นกัน
            if depth_frame is None or color_frame is None:
                continue
            
            # Convert color frame to numpy array
            color_img = np.asanyarray(color_frame.get_data())
            
            # Calculate FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if prev_time > 0 else 0.0
            prev_time = now
            
            # Process frame with detections and tracking
            processed_img = process_frame(
                color_img, 
                depth_frame, 
                detector, 
                tracker, 
                depth_processor,
                yolo_enabled
            )

            if tracker.is_tracking():
                state = tracker.get_state()
                xyz = state.get('xyz') or state.get('xyz_coordinates') 
                confidence = state.get('confidence', state.get('conf', 0.0))
                h_angle = state.get('h_angle', getattr(tracker, 'h_angle', 0.0))
                v_height = state.get('v_height', getattr(tracker, 'v_height', 0.0))

                spear_name = getattr(tracker, 'current_spear_type', 'unknown')
                spear_id = spear_map.get(spear_name, -1.0)

                if xyz is not None:
                    print(f"✅ ส่งข้อมูล: ID={spear_id} ({spear_name}), X={xyz[0]:.2f} , Y={xyz[1]:.2f} , Z={xyz[2]:.2f} , Conf={confidence:.2f} ")
                    
                    # ส่ง spear_id เข้าไปแทนตำแหน่งของ spear_type ในฟังก์ชัน
                    ros_node.publish_target(spear_id, xyz[0], xyz[1], xyz[2], confidence, h_angle, v_height)
            
            rclpy.spin_once(ros_node, timeout_sec=0.001)
            
            # Create depth colormap with actual depth range calculation
            depth_colormap, actual_min_depth, actual_max_depth = create_depth_colormap(depth_frame)
            
            # Get depth at center of frame for crosshair display
            h, w = color_img.shape[:2]
            center_x, center_y = w // 2, h // 2
            center_depth = get_depth_median(depth_frame, center_x, center_y)
            
            # Draw info panel on processed image
            draw_info_panel(processed_img, fps, tracker.get_state())
            
            # Create split view: color+detections on left, depth on right with scale
            # Use actual calculated depth range from the scene
            display_img = create_split_view(
                processed_img, 
                depth_colormap, 
                center_depth=center_depth,
                depth_range=(actual_min_depth, actual_max_depth)
            )
            
            # Display
            cv2.imshow(WIN_NAME, display_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("m") or key == ord("M"):  # กด M เพื่อเปิด/ปิด โมเดล
                yolo_enabled = not yolo_enabled
                state_text = "ON" if yolo_enabled else "OFF"
                print(f"YOLO Model turned {state_text}")
    
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


if __name__ == "__main__":
    main()