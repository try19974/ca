"""
Configuration settings for RealSense YOLO Object Detection System
"""

# YOLO Model Configuration
MODEL_PATH = "/home/korn/Documents/GitHub/fibox_abu_2026_R2_robot/ros2_ws/src/vision_r2/vision_r2/best.pt"
CONF_THRES = 0.70  # Model Confidence Threshold

# RealSense Camera Configuration
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 270
CAMERA_FPS = 15
#
# Depth Processing Parameters
DEPTH_WIN = 5  # Window size for median depth calculation
MIN_VALID_DEPTH = 0.05  # Minimum valid depth in meters (5cm)
MAX_VALID_DEPTH = 10.0  # Maximum valid depth in meters

# Tracking Parameters
SMOOTH_ALPHA = 0.25  # Exponential moving average smoothing factor
HOLD_FRAMES = 8  # Number of frames to hold tracking after object is lost

# Display Configuration
WIN_NAME = "RealSense YOLO (Stable Depth)"
FULLSCREEN = True

# Color scheme for different object classes
COLORS = {
    "fist": (0, 0, 255),      # Red
    "hand": (0, 255, 0),       # Green
    "spear": (0, 165, 255),    # Orange
    "default": (255, 255, 0),  # Cyan
}

# UI Colors
UI_BG_COLOR = (40, 40, 40)
UI_TEXT_COLOR = (255, 255, 255)
TRACKING_COLOR = (0, 255, 0)     # Green when tracking
HOLDING_COLOR = (0, 255, 255)     # Yellow when holding