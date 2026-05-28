"""
Visualization utilities for drawing on images
"""
import cv2
import numpy as np
from vision_r2.config import UI_BG_COLOR, UI_TEXT_COLOR


def clamp(v, lo, hi):
    """Clamp value between min and max bounds"""
    return max(lo, min(hi, v))


def draw_text_bg(
    img, text, x, y, bg=UI_BG_COLOR, fg=UI_TEXT_COLOR, scale=0.6, thick=2, pad=6
):
    """
    Draw text with background rectangle
    
    Args:
        img: Image to draw on
        text: Text to display
        x, y: Position (top-left corner)
        bg: Background color (BGR)
        fg: Foreground/text color (BGR)
        scale: Font scale
        thick: Font thickness
        pad: Padding around text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)

    x1 = x
    y1 = y - th - pad
    x2 = x + tw + pad * 2
    y2 = y + base + pad

    x1 = clamp(x1, 0, img.shape[1] - 1)
    y1 = clamp(y1, 0, img.shape[0] - 1)
    x2 = clamp(x2, 0, img.shape[1] - 1)
    y2 = clamp(y2, 0, img.shape[0] - 1)

    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(img, text, (x + pad, y), font, scale, fg, thick, cv2.LINE_AA)


def draw_detection_box(img, x1, y1, x2, y2, label, color, conf):
    """
    Draw bounding box with label for detected object
    
    Args:
        img: Image to draw on
        x1, y1, x2, y2: Bounding box coordinates
        label: Class label
        color: Box color (BGR)
        conf: Confidence score
    """
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw center point
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
    
    # Draw label
    draw_text_bg(img, label, x1, max(20, y1), bg=color)


def draw_tracking_indicator(img, cx, cy, is_found, color):
    """
    Draw tracking indicator at object center
    
    Args:
        img: Image to draw on
        cx, cy: Center coordinates
        is_found: Whether object is currently detected
        color: Indicator color
    """
    cv2.circle(img, (cx, cy), 7, color, -1)
    cv2.circle(img, (cx, cy), 14, color, 2)


def draw_info_panel(img, fps, spear_state):
    """
    Draw information panel with FPS and tracking data
    
    Args:
        img: Image to draw on
        fps: Current FPS
        spear_state: Dictionary containing spear tracking state
    """
    y_offset = 30
    
    # FPS
    draw_text_bg(img, f"FPS: {fps:.1f}", 10, y_offset, bg=(0, 0, 0))
    y_offset += 35
    
    # Spear tracking info
    if spear_state["cx"] is not None:
        if spear_state["depth"] is None:
            depth_str = "-- m"
        else:
            depth_str = f"{spear_state['depth']:.2f} m"
        
        status = f"SPEAR | conf {spear_state['conf']:.2f} | {depth_str} | lost {spear_state['lost_count']}"
        draw_text_bg(img, status, 10, y_offset, bg=(40, 40, 40))
        y_offset += 35
        
        # Angle and Height
        angle_str = f"H-Angle: {spear_state['h_angle']:+.2f}°"
        draw_text_bg(img, angle_str, 10, y_offset, bg=(40, 40, 40))
        y_offset += 35
        
        height_str = f"V-Height: {spear_state['v_height']:+.2f} m"
        draw_text_bg(img, height_str, 10, y_offset, bg=(40, 40, 40))
        y_offset += 35
        
        # XYZ coordinates if available
        if spear_state["xyz"] is not None:
            x, y, z = spear_state["xyz"]
            xyz_str = f"XYZ: ({x:.2f}, {y:.2f}, {z:.2f}) m"
            draw_text_bg(img, xyz_str, 10, y_offset, bg=(40, 40, 40))


def draw_depth_scale(img, x, y, min_depth, max_depth, width=30, height=200):
    """
    Draw a color scale showing depth value mapping
    
    Args:
        img: Image to draw on
        x, y: Top-left position of the scale
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters
        width: Width of the scale bar
        height: Height of the scale bar
    """
    # Create gradient from blue (close) to red (far)
    for i in range(height):
        # Calculate depth for this position
        ratio = i / height
        depth = min_depth + (max_depth - min_depth) * ratio
        
        # Create color gradient (JET colormap approximation)
        # Blue -> Cyan -> Green -> Yellow -> Red
        if ratio < 0.25:
            # Blue to Cyan
            r = 0
            g = int(255 * (ratio / 0.25))
            b = 255
        elif ratio < 0.5:
            # Cyan to Green
            r = 0
            g = 255
            b = int(255 * (1 - (ratio - 0.25) / 0.25))
        elif ratio < 0.75:
            # Green to Yellow
            r = int(255 * ((ratio - 0.5) / 0.25))
            g = 255
            b = 0
        else:
            # Yellow to Red
            r = 255
            g = int(255 * (1 - (ratio - 0.75) / 0.25))
            b = 0
        
        color = (b, g, r)  # BGR format
        cv2.rectangle(img, (x, y + i), (x + width, y + i + 1), color, -1)
    
    # Draw border
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 2)
    
    # Add depth labels
    label_x = x + width + 5
    
    # Top (min depth)
    draw_text_bg(img, f"{min_depth:.1f}m", label_x, y + 15, bg=(0, 0, 0), scale=0.5)
    
    # Middle
    mid_depth = (min_depth + max_depth) / 2
    draw_text_bg(img, f"{mid_depth:.1f}m", label_x, y + height // 2 + 5, bg=(0, 0, 0), scale=0.5)
    
    # Bottom (max depth)
    draw_text_bg(img, f"{max_depth:.1f}m", label_x, y + height - 5, bg=(0, 0, 0), scale=0.5)
    
    # Title
    draw_text_bg(img, "DEPTH", x - 5, y - 10, bg=(0, 0, 0), scale=0.5)


def draw_center_crosshair(img, center_depth=None):
    """
    Draw crosshair at image center with depth value if available
    
    Args:
        img: Image to draw on
        center_depth: Depth at center in meters (None if not available)
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Draw crosshair
    line_len = 20
    color = (0, 255, 255)  # Yellow
    thickness = 2
    
    # Horizontal line
    cv2.line(img, (cx - line_len, cy), (cx + line_len, cy), color, thickness)
    # Vertical line
    cv2.line(img, (cx, cy - line_len), (cx, cy + line_len), color, thickness)
    
    # Draw center circle
    cv2.circle(img, (cx, cy), 3, color, -1)
    
    # Display depth if available
    if center_depth is not None and center_depth > 0:
        label = f"{center_depth:.2f}m"
        # Position label below crosshair
        draw_text_bg(img, label, cx - 30, cy + 30, bg=(0, 0, 0), fg=(0, 255, 255), scale=0.6)


def create_split_view(color_img, depth_colormap, center_depth=None, depth_range=(0.0, 10.0)):
    """
    Create side-by-side view of color and depth images with depth scale
    
    Args:
        color_img: Color image
        depth_colormap: Colorized depth map
        center_depth: Depth at image center in meters (optional)
        depth_range: Tuple of (min_depth, max_depth) for scale
    
    Returns:
        Combined image with color on left, depth on right, with depth scale
    """
    # Ensure both images have the same height
    h = max(color_img.shape[0], depth_colormap.shape[0])
    
    # Resize if needed
    if color_img.shape[0] != h:
        color_img = cv2.resize(color_img, (color_img.shape[1], h))
    if depth_colormap.shape[0] != h:
        depth_colormap = cv2.resize(depth_colormap, (depth_colormap.shape[1], h))
    
    # Draw crosshair on depth map
    draw_center_crosshair(depth_colormap, center_depth)
    
    # Add margin on right side for depth scale
    margin = 120
    depth_with_margin = np.zeros((h, depth_colormap.shape[1] + margin, 3), dtype=np.uint8)
    depth_with_margin[:, :depth_colormap.shape[1]] = depth_colormap
    
    # Draw depth scale on the margin
    scale_x = depth_colormap.shape[1] + 20
    scale_y = 60
    min_depth, max_depth = depth_range
    draw_depth_scale(depth_with_margin, scale_x, scale_y, min_depth, max_depth, width=30, height=h - 120)
    
    # Combine horizontally
    combined = np.hstack((color_img, depth_with_margin))
    
    # Add labels
    draw_text_bg(combined, "COLOR + DETECTIONS", 10, 30, bg=(0, 100, 0))
    draw_text_bg(
        combined, 
        "DEPTH MAP", 
        color_img.shape[1] + 10, 
        30, 
        bg=(100, 0, 0)
    )
    
    return combined