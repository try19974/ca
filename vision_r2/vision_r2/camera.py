"""
RealSense camera initialization and management
"""
import pyrealsense2 as rs
from vision_r2.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class RealSenseCamera:
    """
    Wrapper for RealSense camera operations
    """
    
    def __init__(self, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        """
        Initialize RealSense camera
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        self._configure()
    
    def _configure(self):
        """Configure camera streams"""
        self.config.enable_stream(
            rs.stream.depth, 
            self.width, 
            self.height, 
            rs.format.z16, 
            self.fps
        )
        self.config.enable_stream(
            rs.stream.color, 
            self.width, 
            self.height, 
            rs.format.bgr8, 
            self.fps
        )
    
    def start(self):
        """Start the camera pipeline"""
        print("Opening RealSense camera...")
        self.pipeline.start(self.config)
        
        # Create alignment object to align depth to color
        self.align = rs.align(rs.stream.color)
        print("Camera started successfully")
    
    def get_frames(self):
        """
        Get aligned depth and color frames
        
        Returns:
            Tuple of (depth_frame, color_frame, depth_colormap)
            Returns (None, None, None) if frames are not available
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None, None
            
            return depth_frame, color_frame
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None, None
    
    def stop(self):
        """Stop the camera pipeline"""
        print("Stopping camera...")
        self.pipeline.stop()
        print("Camera stopped")