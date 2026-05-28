"""
Object tracking and state management
"""
from vision_r2.config import SMOOTH_ALPHA, HOLD_FRAMES


class ObjectTracker:
    """
    Track an object across frames with smoothing and state management
    """
    
    def __init__(self, object_name="object"):
        """
        Initialize tracker
        
        Args:
            object_name: Name of the object being tracked
        """
        self.object_name = object_name
        self.reset()
    
    def reset(self):
        """Reset all tracking state"""
        self.found_this_frame = False
        self.lost_count = 0
        self.cx = None
        self.cy = None
        self.depth = None
        self.xyz = None
        self.conf = 0.0
        self.h_angle = 0.0
        self.v_height = 0.0
    
    def ema(self, old, new, alpha=SMOOTH_ALPHA):
        """
        Exponential moving average for smoothing
        
        Args:
            old: Previous value
            new: New value
            alpha: Smoothing factor (0-1)
        
        Returns:
            Smoothed value
        """
        return (1 - alpha) * old + alpha * new
    
    def update_found(self, cx, cy, depth_m, xyz, conf, h_angle=0.0, v_height=0.0):
        """
        Update tracker when object is detected
        
        Args:
            cx, cy: Center coordinates
            depth_m: Depth in meters
            xyz: 3D coordinates tuple
            conf: Confidence score
            h_angle: Horizontal angle offset
            v_height: Vertical height offset
        """
        if self.cx is None:
            # First detection - initialize without smoothing
            self.cx = cx
            self.cy = cy
            self.depth = depth_m
            self.xyz = xyz
        else:
            # Apply smoothing to existing track
            self.cx = int(self.ema(self.cx, cx))
            self.cy = int(self.ema(self.cy, cy))
            
            if depth_m is not None:
                if self.depth is None:
                    self.depth = depth_m
                else:
                    self.depth = self.ema(self.depth, depth_m)
                self.xyz = xyz
        
        self.conf = conf
        self.h_angle = h_angle
        self.v_height = v_height
        self.found_this_frame = True
        self.lost_count = 0
    
    def update_lost(self):
        """Update tracker when object is not detected"""
        self.found_this_frame = False
        self.lost_count += 1
        
        if self.lost_count > HOLD_FRAMES:
            self.reset()
    
    def get_state(self):
        """
        Get current tracking state as dictionary
        
        Returns:
            Dictionary with all tracking parameters
        """
        return {
            "found_this_frame": self.found_this_frame,
            "lost_count": self.lost_count,
            "cx": self.cx,
            "cy": self.cy,
            "depth": self.depth,
            "xyz": self.xyz,
            "conf": self.conf,
            "h_angle": self.h_angle,
            "v_height": self.v_height,
        }
    
    def is_tracking(self):
        """Check if currently tracking an object"""
        return self.cx is not None and self.cy is not None