"""
Geometry calculations for object positioning
"""
import numpy as np


def calculate_offsets(cx, cy, img_w, img_h, depth_m):
    """
    Calculate horizontal angle and vertical height offsets for an object
    
    Args:
        cx: Center X pixel coordinate
        cy: Center Y pixel coordinate
        img_w: Image width in pixels
        img_h: Image height in pixels
        depth_m: Depth to object in meters
    
    Returns:
        Tuple of (horizontal_angle_degrees, vertical_height_meters)
    """
    # Typical RealSense D435 has ~69° horizontal FOV and ~42° vertical FOV
    # These can be adjusted based on your specific camera model
    HFOV = 69.0  # Horizontal field of view in degrees
    VFOV = 42.0  # Vertical field of view in degrees
    
    # Calculate horizontal angle offset
    # Positive = object is to the right of center
    # Negative = object is to the left of center
    center_x = img_w / 2.0
    pixel_offset_x = cx - center_x
    degrees_per_pixel_h = HFOV / img_w
    h_angle = pixel_offset_x * degrees_per_pixel_h
    
    # Calculate vertical height offset
    # This estimates the real-world height difference from camera center
    center_y = img_h / 2.0
    pixel_offset_y = center_y - cy  # Inverted because Y increases downward in image
    
    if depth_m is not None and depth_m > 0:
        # Convert pixel offset to angle
        degrees_per_pixel_v = VFOV / img_h
        v_angle = pixel_offset_y * degrees_per_pixel_v
        
        # Convert angle to height using depth
        # height = depth * tan(angle)
        v_angle_rad = np.radians(v_angle)
        v_height = depth_m * np.tan(v_angle_rad)
    else:
        v_height = 0.0
    
    return h_angle, v_height