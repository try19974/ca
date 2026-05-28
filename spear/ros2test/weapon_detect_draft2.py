"""
Weapon detection with XY-centroid depth — RealSense version (ROS2 Node).

Detects weapons/objects and publishes centroid positions and depth information.
Uses configuration from centroid_config.txt

Output per detection:
  index, class, x_from_principle(+R/-L)mm, y_from_principle(+D/-U)mm,
  z_depth_mm, abs_dist_mm
"""

import json
import math
import time
from itertools import product as iproduct

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point


# ── Config ────────────────────────────────────────────────────────────────
CENTROID_FILE = "centroid_config.txt"
VIDEO_WIN     = "Weapon Detect Draft2"
SIDE_W        = 310

MODEL  = "yolonew.onnx"
CONF   = 0.5
IMGSZ  = 640
WIDTH  = 640
HEIGHT = 480
FPS    = 30

max_depth_mm      = 5000
max_bbox_area_pct = 50.0
horiz_align_pct   = 10.0
seg_depth_dev_pct = 35.0
depth_tol_mm      = 150

class_cent_x_pct  = {}
class_cent_y_pct  = {}

LAYOUT = ["spearhead", "fist", "hand", "hand", "fist", "spearhead"]

_json_file = None


# ── Config loading ────────────────────────────────────────────────────────

def load_config(class_names):
    global max_depth_mm, max_bbox_area_pct, horiz_align_pct
    global seg_depth_dev_pct, depth_tol_mm
    try:
        data = {}
        with open(CENTROID_FILE) as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = float(v.strip())

        if "max_depth_mm"      in data: max_depth_mm      = int(data["max_depth_mm"])
        if "max_bbox_area_pct" in data: max_bbox_area_pct = data["max_bbox_area_pct"]
        if "horiz_align_pct"   in data: horiz_align_pct   = data["horiz_align_pct"]
        if "seg_depth_dev_pct" in data: seg_depth_dev_pct = data["seg_depth_dev_pct"]
        if "depth_tol_mm"      in data: depth_tol_mm      = int(data["depth_tol_mm"])

        for idx, name in class_names.items():
            if f"{name}_cent_x" in data:
                class_cent_x_pct[idx] = data[f"{name}_cent_x"]
            if f"{name}_cent_y" in data:
                class_cent_y_pct[idx] = data[f"{name}_cent_y"]

        print(f"Loaded {CENTROID_FILE}")
        print(f"  depth_max={max_depth_mm}mm  bbox_max={max_bbox_area_pct:.1f}%"
              f"  depth_tol={depth_tol_mm}mm")
        for idx, name in class_names.items():
            print(f"  {name}: centX={class_cent_x_pct.get(idx,50.0):.1f}%"
                  f"  centY={class_cent_y_pct.get(idx,50.0):.1f}%")
    except FileNotFoundError:
        print(f"{CENTROID_FILE} not found — using defaults (centX=50% centY=50%)")


# ── Filters ───────────────────────────────────────────────────────────────

def filter_horiz_align(raw_boxes, h_f):
    if len(raw_boxes) <= 1:
        return raw_boxes
    cy_vals   = [(b["y1"] + b["y2"]) // 2 for b in raw_boxes]
    median_cy = float(np.median(cy_vals))
    max_dev   = horiz_align_pct / 100.0 * h_f
    return [b for b in raw_boxes
            if abs((b["y1"] + b["y2"]) / 2 - median_cy) <= max_dev]


def filter_outlier_boxes(raw_boxes):
    if len(raw_boxes) <= 1:
        return raw_boxes
    z_vals   = [b["z_mm"] for b in raw_boxes
                if b["z_mm"] is not None and b["z_mm"] > 0]
    if not z_vals:
        return raw_boxes
    median_z = float(np.median(z_vals))
    thr      = seg_depth_dev_pct / 100.0
    return [b for b in raw_boxes
            if b["z_mm"] is not None and b["z_mm"] > 0 and
            abs(b["z_mm"] - median_z) / median_z <= thr]


# ── Placement index assignment ────────────────────────────────────────────

def assign_placement_indices(detections, frame_width):
    n = len(detections)
    if n == 0:
        return {}

    xs           = [(d["bbox"][0] + d["bbox"][2]) / 2 for d in detections]
    prefer_right = (sum(xs) / n) > frame_width / 2

    valid_for = []
    for d in detections:
        cls   = d["class_name"]
        valid = [i for i, c in enumerate(LAYOUT) if c == cls]
        valid_for.append(valid if valid else [-1])

    best_key   = None
    best_combo = [v[0] for v in valid_for]

    for combo in iproduct(*valid_for):
        known = [c for c in combo if c != -1]
        if len(set(known)) != len(known):
            continue
        score = 0
        for a in range(n):
            for b in range(a + 1, n):
                ca, cb = combo[a], combo[b]
                if ca == -1 or cb == -1:
                    continue
                if (xs[a] < xs[b]) == (ca < cb):
                    score += 1
                else:
                    score -= 1
        idx_sum  = sum(c for c in combo if c != -1)
        tiebreak = idx_sum if prefer_right else -idx_sum
        key = (score, tiebreak)
        if best_key is None or key > best_key:
            best_key   = key
            best_combo = list(combo)

    return {i: best_combo[i] for i in range(n) if best_combo[i] != -1}


# ── Detection output ──────────────────────────────────────────────────────

def on_detection(node, detections):
    for d in detections:
        record = {
            "index": d["index"],
            "class": d["class_name"],
            "x_from_principle_mm": d["horiz_mm"],
            "y_from_principle_mm": d["vert_mm"],
            "z_depth_mm": d["z_mm"],
            "abs_dist_mm": d["abs_dist_mm"],
            "confidence": d["conf"],
        }
        node.publish_detection(record)


# ── Sidebar ───────────────────────────────────────────────────────────────

def draw_sidebar(detections, h, cam_ref_px):
    panel = np.full((h, SIDE_W, 3), 22, dtype=np.uint8)

    def t(s, y, col=(190, 190, 190), scale=0.42, thick=1):
        cv2.putText(panel, s, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

    y = 22
    t("Detections", y, (130, 130, 130), 0.50)
    y += 16

    t("Origin (0,0 mm XY):", y, (100, 180, 100), 0.38)
    y += 14
    t(f"  principal_ctr = ({cam_ref_px[0]}, {cam_ref_px[1]}) px",
      y, (100, 180, 100), 0.36)
    y += 13
    t("  (where optical axis hits sensor)", y, (70, 130, 70), 0.33)
    y += 14
    t("Axes from that origin:", y, (150, 150, 150), 0.36)
    y += 13
    t("  X: <-(-X)  [0]  (+X)->", y, (100, 200, 255), 0.36)
    y += 13
    t("  Y:  ^(-Y)  [0]  (+Y)v ", y, (100, 200, 255), 0.36)
    y += 13
    t("  Z: fwd into scene (depth)", y, (100, 200, 255), 0.36)
    y += 8
    cv2.line(panel, (8, y), (SIDE_W - 8, y), (55, 55, 55), 1)
    y += 18

    if not detections:
        t("(none)", y, (70, 70, 70))
        return panel

    for d in detections:
        c        = d["centroid_px"]
        cent_str = f"({c[0]},{c[1]})px" if c else "none"
        z_str    = f"{d['z_mm']}mm"          if d["z_mm"]        is not None else "-"
        abs_str  = f"{d['abs_dist_mm']}mm"   if d["abs_dist_mm"] is not None else "-"
        h_str    = f"{d['horiz_mm']:+.1f}mm" if d["horiz_mm"]    is not None else "-"
        v_str    = f"{d['vert_mm']:+.1f}mm"  if d["vert_mm"]     is not None else "-"

        t(f"[{d['index']}] {d['class_name']}  {d['conf']:.2f}", y, (0, 210, 255), 0.46)
        y += 18
        t(f"  centroid:{cent_str}", y, (180, 255, 180), 0.40)
        y += 15
        t(f"  z(straight):{z_str}", y, (80, 220, 180), 0.38)
        y += 15
        t(f"  abs_dist:   {abs_str}", y, (80, 255, 160), 0.38)
        y += 15
        t(f"  X(H):{h_str}  Y(V):{v_str}", y, (160, 220, 100), 0.38)
        y += 15
        cv2.line(panel, (8, y), (SIDE_W - 8, y), (45, 45, 45), 1)
        y += 10

        if y > h - 30:
            break

    return panel


# ── ROS2 Node ────────────────────────────────────────────────────────────

class WeaponDetectorNode(Node):
    def __init__(self):
        super().__init__('weapon_detector')
        
        self.get_logger().info("Initializing Weapon Detector Node...")
        
        # Initialize model
        self.model = YOLO(MODEL)
        self.class_names = self.model.names
        self.get_logger().info(f"Model: {MODEL} | Classes: {self.class_names}")
        
        for idx in self.class_names:
            class_cent_x_pct[idx] = 50.0
            class_cent_y_pct[idx] = 50.0
        
        load_config(self.class_names)
        
        np.random.seed(42)
        self.colors = {idx: tuple(int(c) for c in np.random.randint(80, 255, 3))
                       for idx in self.class_names}
        
        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.get_logger().info(f"RealSense: {WIDTH}x{HEIGHT} @ {FPS}fps")
        
        intr = (self.pipeline.get_active_profile()
                .get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics())
        self.fx, self.fy, self.ppx, self.ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
        self.get_logger().info(f"Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} cx={self.ppx:.1f} cy={self.ppy:.1f}")
        self.cam_ref_px = (int(self.ppx), int(self.ppy))
        
        # Warm up camera
        for _ in range(30):
            try:
                self.pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError:
                pass
        
        # Setup display window
        cv2.namedWindow(VIDEO_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VIDEO_WIN, WIDTH + SIDE_W, HEIGHT)
        
        # Publishers
        self.detection_pub = self.create_publisher(String, 'weapon_detections', 10)
        self.point_pub = self.create_publisher(Point, 'weapon_point', 10)
        
        # JSON logging
        self._json_file = open("detections.json", "w", encoding="utf-8")
        self.get_logger().info("Logging detections -> detections.json")
        
        # FPS tracking
        self.fps_smooth = 0.0
        self.alpha = 0.1
        
        # Create timer for main loop
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30 Hz
    
    def publish_detection(self, record):
        """Publish detection data to ROS2 topics"""
        msg = String()
        msg.data = json.dumps(record)
        self.detection_pub.publish(msg)
        
        # Also publish as Point for single coordinate
        point_msg = Point()
        point_msg.x = float(record.get("x_from_principle_mm", 0.0))
        point_msg.y = float(record.get("y_from_principle_mm", 0.0))
        point_msg.z = float(record.get("z_depth_mm", 0.0))
        self.point_pub.publish(point_msg)
        
        # Log to file
        if self._json_file is not None:
            self._json_file.write(json.dumps(record) + "\n")
            self._json_file.flush()
        
        self.get_logger().info(f"Detection: {record}")
    
    def process_frame(self):
        """Main processing loop - called by timer"""
        try:
            t0 = time.perf_counter()
            
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                self.get_logger().error(f"Camera error: {e}")
                return
            
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                return
            
            frame = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())
            h_f, w_f = frame.shape[:2]
            display = frame.copy()
            
            results = self.model.predict(frame, imgsz=IMGSZ, conf=CONF,
                                        device="cpu", verbose=False)
            
            # ── Pass 1: collect + bbox area filter
            frame_area = w_f * h_f
            raw_boxes = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf_v = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, str(cls_id))
                    color = self.colors.get(cls_id, (0, 255, 0))
                    
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w_f, x2); y2 = min(h_f, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2 - x1) * (y2 - y1) / frame_area * 100 > max_bbox_area_pct:
                        continue
                    
                    cx_pct = class_cent_x_pct.get(cls_id, 50.0)
                    cy_pct = class_cent_y_pct.get(cls_id, 50.0)
                    scx = int(x1 + (x2 - x1) * cx_pct / 100.0)
                    scy = int(y1 + (y2 - y1) * cy_pct / 100.0)
                    scx = max(x1, min(x2 - 1, scx))
                    scy = max(y1, min(y2 - 1, scy))
                    
                    z_mm = horiz_mm = vert_mm = abs_dist_mm = None
                    pad = 3
                    patch = depth[max(0, scy-pad): min(h_f, scy+pad+1),
                                  max(0, scx-pad): min(w_f, scx+pad+1)]
                    valid = patch[(patch > 0) & (patch <= max_depth_mm)]
                    if valid.size > 0:
                        obj_d = float(np.median(valid))
                        if depth_tol_mm > 0:
                            band = valid[np.abs(valid.astype(float) - obj_d) <= depth_tol_mm]
                            if band.size > 0:
                                obj_d = float(np.median(band))
                        z_mm = int(obj_d)
                        horiz_mm = round((scx - self.ppx) * obj_d / self.fx, 1)
                        vert_mm = round((scy - self.ppy) * obj_d / self.fy, 1)
                        abs_dist_mm = round(
                            math.sqrt(horiz_mm**2 + vert_mm**2 + obj_d**2), 1)
                    
                    raw_boxes.append({
                        "bbox": (x1, y1, x2, y2),
                        "class_name": cls_name,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf_v, "cls_id": cls_id, "color": color,
                        "centroid_px": (scx, scy),
                        "z_mm": z_mm, "horiz_mm": horiz_mm,
                        "vert_mm": vert_mm, "abs_dist_mm": abs_dist_mm,
                    })
            
            # ── Pass 2: alignment + outlier filters
            raw_boxes = filter_horiz_align(raw_boxes, h_f)
            raw_boxes = filter_outlier_boxes(raw_boxes)
            
            # ── Pass 3: placement indices
            placement = assign_placement_indices(raw_boxes, w_f)
            detections = []
            for i, rb in enumerate(raw_boxes):
                detections.append({
                    "index": placement.get(i, -1),
                    "class_name": rb["class_name"],
                    "conf": rb["conf"],
                    "bbox": rb["bbox"],
                    "centroid_px": rb["centroid_px"],
                    "cam_ref_px": self.cam_ref_px,
                    "z_mm": rb["z_mm"],
                    "horiz_mm": rb["horiz_mm"],
                    "vert_mm": rb["vert_mm"],
                    "abs_dist_mm": rb["abs_dist_mm"],
                    "color": rb["color"],
                })
            
            # ── Pass 4: draw
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                scx, scy = d["centroid_px"]
                color = d["color"]
                
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.line(display, (scx, y1), (scx, y2), (180, 255, 180), 1)
                cv2.line(display, (x1, scy), (x2, scy), (180, 255, 180), 1)
                cv2.drawMarker(display, (scx, scy), (0, 255, 255),
                               cv2.MARKER_CROSS, 14, 2)
                
                if d["z_mm"] is not None:
                    ann = f"abs:{d['abs_dist_mm']}mm  z:{d['z_mm']}mm"
                    cv2.putText(display, ann, (scx + 8, scy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
                
                lbl = f"[{d['index']}] {d['class_name']} {d['conf']:.2f}"
                (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display, (x1, y1 - th - bl - 4), (x1 + tw, y1), color, -1)
                cv2.putText(display, lbl, (x1, y1 - bl - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            
            cv2.drawMarker(display, self.cam_ref_px, (100, 180, 100),
                           cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
            
            # ── Publish detections
            if detections:
                on_detection(self, detections)
            
            dt = time.perf_counter() - t0
            self.fps_smooth = self.alpha * (1.0 / max(dt, 1e-6)) + (1 - self.alpha) * self.fps_smooth
            cv2.putText(display, f"FPS: {self.fps_smooth:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            composite = np.hstack([display, draw_sidebar(detections, h_f, self.cam_ref_px)])
            cv2.imshow(VIDEO_WIN, composite)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.destroy_node()
            elif key == ord("s"):
                fname = f"capture_{int(time.time())}.png"
                cv2.imwrite(fname, composite)
                self.get_logger().info(f"Screenshot: {fname}")
        
        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {e}")
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        try:
            self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        
        if self._json_file is not None:
            self._json_file.close()
            self.get_logger().info("Detections saved -> detections.json")
        
        self.get_logger().info("Weapon Detector Node stopped.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WeaponDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
