# Weapon Detection — Interface Specification

This document describes the input and output interface of the weapon detection node for ROS2 integration.

---

## Input

### Camera

| Parameter | Value |
|---|---|
| Device | Intel RealSense D-series (USB 3) |
| Colour stream | 640 × 480 px, BGR8, 30 fps |
| Depth stream | 640 × 480 px, Z16 (mm), 30 fps |
| Depth alignment | Depth frame is aligned to colour frame — every colour pixel has a matching depth value |

### Camera Intrinsics (read at runtime from the device)

| Parameter | Symbol | Unit |
|---|---|---|
| Focal length X | `fx` | pixels |
| Focal length Y | `fy` | pixels |
| Principal point X | `ppx` | pixels |
| Principal point Y | `ppy` | pixels |

Intrinsics are read automatically via `get_intrinsics()` and printed at startup:
```
Intrinsics: fx=614.1 fy=614.4 cx=319.4 cy=253.7
```

### Model

| Parameter | Value |
|---|---|
| Format | ONNX (YOLOv8 detection) |
| Input size | 640 × 640 px |
| Confidence threshold | 0.5 |
| Classes | 0: weapondetect, 1: fist, 2: hand, 3: spearhead |

---

## Coordinate System

Origin `(0, 0, 0)` is the camera optical centre (principal point).

```
        Z+ (forward, depth into scene)
       /
      /
     +————————> X+ (right)
     |
     |
     v
    Y+ (down)
```

| Axis | Direction | Unit |
|---|---|---|
| X | + = right of camera, − = left | mm |
| Y | + = below camera centre, − = above | mm |
| Z | + = forward into the scene (depth) | mm |

The principal point pixel `(ppx, ppy)` is where X=0 and Y=0.  
It is **not** exactly the image centre — it comes from the camera's factory calibration.

### Pixel → mm Conversion

```
x_mm = (centroid_pixel_x - ppx) * z_mm / fx
y_mm = (centroid_pixel_y - ppy) * z_mm / fy
abs_dist_mm = sqrt(x_mm² + y_mm² + z_mm²)
```

---

## Output

### File

`detections.json` — written in the working directory, overwritten each run.  
Format: **JSON Lines** (one JSON object per line, one line per detection per frame).

### JSON Object Schema

```json
{
  "index":                <int>,
  "class":                <string>,
  "x_from_principle_mm":  <float | null>,
  "y_from_principle_mm":  <float | null>,
  "z_depth_mm":           <int | null>,
  "abs_dist_mm":          <float | null>
}
```

### Field Definitions

| Field | Type | Unit | Description |
|---|---|---|---|
| `index` | int | — | Placement slot 0–5. −1 if class is not in the layout. |
| `class` | string | — | Detected weapon class name |
| `x_from_principle_mm` | float or null | mm | Horizontal offset from principal point. Positive = right, negative = left |
| `y_from_principle_mm` | float or null | mm | Vertical offset from principal point. Positive = down, negative = up |
| `z_depth_mm` | int or null | mm | Straight depth along camera Z-axis (not Euclidean distance) |
| `abs_dist_mm` | float or null | mm | True 3-D Euclidean distance from camera origin |

All position fields are `null` when depth data is unavailable at the centroid pixel.

### Example Output

```json
{"index": 0, "class": "spearhead", "x_from_principle_mm": -134.2, "y_from_principle_mm": 18.5, "z_depth_mm": 920, "abs_dist_mm": 931.0}
{"index": 2, "class": "hand",      "x_from_principle_mm": 12.3,   "y_from_principle_mm": -5.6, "z_depth_mm": 850, "abs_dist_mm": 850.1}
```

### Placement Index Layout

Weapons are assigned to fixed left-to-right slots across the frame:

```
index:    0          1       2    |    3       4        5
class:  spearhead  fist   hand   |  hand    fist   spearhead
          (left side)            |         (right side)
```

Assignment is determined by matching each detection's horizontal screen position to the layout order that best preserves left-to-right consistency.

---

## Timing

- Output rate: up to 30 Hz (one set of lines per frame, only when detections exist)
- Each frame may produce 0–6 lines depending on detections
- All detections in a frame are written sequentially with no frame-level wrapper

---

## Filtering Applied Before Output

| Filter | Default | Effect |
|---|---|---|
| Max bbox area | 50% of frame | Drops false positives that fill most of the frame |
| Horizontal alignment | 10% of frame height | Drops boxes whose centre Y is far from the group median — assumes all weapons appear on roughly the same horizontal line |
| Depth deviation | 35% from group median | Drops boxes whose centroid depth is an outlier vs the other detections |
| Depth tolerance band | ±150 mm around object | Rejects depth pixels that belong to the background behind/in front of the weapon |

---

## Notes for ROS2

- The output coordinate frame matches a **camera optical frame** convention: Z forward, X right, Y down — this maps directly to ROS2 `camera_optical_frame` (not `camera_frame` which is Z forward, X left, Y down — note the X sign difference).
- `z_depth_mm` is the depth along the Z-axis only. For 3-D position in a ROS2 `PointStamped`, use `x_mm`, `y_mm`, `z_mm` directly after unit conversion (`/ 1000.0` for metres).
- The JSON Lines stream can be consumed by a ROS2 node reading the file line-by-line, or the `on_detection()` function in `weapon_detect_draft2.py` can be replaced with a ROS2 publisher directly.
