"""
Depth processing utilities for RealSense camera
"""
import numpy as np
import pyrealsense2 as rs
from vision_r2.config import DEPTH_WIN, MIN_VALID_DEPTH


def clamp(v, lo, hi):
    """Clamp value between min and max bounds"""
    return max(lo, min(hi, v))


def get_depth_median(depth_frame, x, y, win=DEPTH_WIN):
    """
    Get median depth value in a window around a pixel
    
    Args:
        depth_frame: RealSense depth frame
        x: X coordinate (center of window)
        y: Y coordinate (center of window)
        win: Window size (default from config)
    
    Returns:
        Median depth value in meters, or None if no valid values
    """
    w = depth_frame.get_width()
    h = depth_frame.get_height()

    half = win // 2
    x1 = clamp(x - half, 0, w - 1)
    x2 = clamp(x + half, 0, w - 1)
    y1 = clamp(y - half, 0, h - 1)
    y2 = clamp(y + half, 0, h - 1)

    vals = []
    for yy in range(y1, y2 + 1):
        for xx in range(x1, x2 + 1):
            d = depth_frame.get_distance(xx, yy)
            if d > 0:
                vals.append(d)

    if not vals:
        return None
    return float(np.median(vals))


def pixel_to_xyz(depth_frame, x, y, depth_m):
    """
    Convert pixel coordinates and depth to 3D XYZ coordinates
    
    Args:
        depth_frame: RealSense depth frame
        x: Pixel x coordinate
        y: Pixel y coordinate
        depth_m: Depth in meters
    
    Returns:
        Tuple of (X, Y, Z) in meters
    """
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth_m)
    return (X, Y, Z)


def create_depth_colormap(depth_frame, min_depth=None, max_depth=None):
    """
    Create a colorized depth map for visualization using actual depth values
    
    Args:
        depth_frame: RealSense depth frame
        min_depth: Minimum depth for color mapping (auto-detect if None)
        max_depth: Maximum depth for color mapping (auto-detect if None)
    
    Returns:
        Tuple of (colorized depth image in BGR format, actual_min_depth, actual_max_depth)
    """
    # Get depth data as numpy array in millimeters, convert to meters
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_scale = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
    
    # Convert to float and scale to meters
    depth_array = depth_image.astype(np.float32) * 0.001  # mm to meters
    
    # Filter out invalid depths
    valid_mask = depth_array > 0.05
    valid_depths = depth_array[valid_mask]
    
    # Auto-detect range if not provided
    if min_depth is None:
        if len(valid_depths) > 0:
            min_depth = float(np.percentile(valid_depths, 5))  # Use 5th percentile
        else:
            min_depth = 0.05
    
    if max_depth is None:
        if len(valid_depths) > 0:
            max_depth = float(np.percentile(valid_depths, 95))  # Use 95th percentile
        else:
            max_depth = 10.0
    
    # Ensure min < max
    if max_depth <= min_depth:
        max_depth = min_depth + 1.0
    
    # Normalize depth to 0-1 range
    normalized = np.clip((depth_array - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    
    # Create RGB channels using vectorized operations
    h, w = depth_array.shape
    colormap = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply JET-like colormap (vectorized)
    # Blue to Cyan (0.0 - 0.25)
    mask1 = normalized < 0.25
    ratio1 = normalized / 0.25
    colormap[mask1, 0] = 255  # B
    colormap[mask1, 1] = (255 * ratio1[mask1]).astype(np.uint8)  # G
    colormap[mask1, 2] = 0  # R
    
    # Cyan to Green (0.25 - 0.5)
    mask2 = (normalized >= 0.25) & (normalized < 0.5)
    ratio2 = (normalized - 0.25) / 0.25
    colormap[mask2, 0] = (255 * (1 - ratio2[mask2])).astype(np.uint8)  # B
    colormap[mask2, 1] = 255  # G
    colormap[mask2, 2] = 0  # R
    
    # Green to Yellow (0.5 - 0.75)
    mask3 = (normalized >= 0.5) & (normalized < 0.75)
    ratio3 = (normalized - 0.5) / 0.25
    colormap[mask3, 0] = 0  # B
    colormap[mask3, 1] = 255  # G
    colormap[mask3, 2] = (255 * ratio3[mask3]).astype(np.uint8)  # R
    
    # Yellow to Red (0.75 - 1.0)
    mask4 = normalized >= 0.75
    ratio4 = (normalized - 0.75) / 0.25
    colormap[mask4, 0] = 0  # B
    colormap[mask4, 1] = (255 * (1 - ratio4[mask4])).astype(np.uint8)  # G
    colormap[mask4, 2] = 255  # R
    
    # Set invalid depths to black
    colormap[~valid_mask] = [0, 0, 0]
    
    return colormap, min_depth, max_depth