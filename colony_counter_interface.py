import sys
import os
import random
import joblib


from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QToolBar,
    QAction,
    QMessageBox,
    QLineEdit,
    QSlider,
)

from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QRectF, QPointF

import pandas as pd
import numpy as np
import cv2
from openpyxl.drawing.image import Image as XLImage


NUM_CIRCLES = 80  # number of petri dishes (circles)

def extract_plate_warp(img_bgr, W=1200, H=800):
    """
    Replicates your Project-2 preprocessing:
    - grayscale -> Otsu -> invert -> morphological close
    - largest contour -> minAreaRect -> perspective warp to (W,H)
    Returns: warped_bgr
    """
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.int32)

    def order_points(pts):
        pts = pts.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],        # top-left
            pts[np.argmin(diff)],     # top-right
            pts[np.argmax(s)],        # bottom-right
            pts[np.argmax(diff)]      # bottom-left
        ], dtype=np.float32)

    src = order_points(box)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H))
    return warped
# ----------------------------
# Well detection (from Project 2)
# ----------------------------
HOUGH_DP, HOUGH_MINDIST = 1.2, 45
HOUGH_PARAM1, HOUGH_PARAM2 = 80, 25
HOUGH_MINR, HOUGH_MAXR = 35, 40
REMOVE_OVERLAP_MINDIST = 60

def detect_wells(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MINDIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MINR, maxRadius=HOUGH_MAXR
    )
    return np.round(circles[0]).astype(int) if circles is not None else None

def remove_overlaps(circles, min_dist=REMOVE_OVERLAP_MINDIST):
    if circles is None:
        return None
    final = []
    for c in circles:
        if all(np.hypot(c[0] - f[0], c[1] - f[1]) >= min_dist for f in final):
            final.append(c)
    return np.array(final, dtype=int) if len(final) else None

# Drop the outermost columns (like training script)
DROP_EDGE_COLS = (1, 12)

def label_wells_A1_H12(circles, row_labels="ABCDEFGH"):
    """Assign approximate A1..H12 based on y-then-x sorting."""
    if circles is None or len(circles) == 0:
        return {}

    circles_sorted = sorted(circles, key=lambda c: c[1])  # sort by y
    rows = [[] for _ in range(8)]
    row_thresh = 35

    for c in circles_sorted:
        y = c[1]
        for row in rows:
            if (len(row) == 0) or (abs(row[0][1] - y) < row_thresh):
                row.append(c)
                break

    for row in rows:
        row.sort(key=lambda c: c[0])  # sort by x within row

    well_map = {}
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            well_map[f"{row_labels[i]}{j+1}"] = (int(c[0]), int(c[1]), int(c[2]))
    return well_map

def remove_edge_columns(well_map, cols=DROP_EDGE_COLS):
    return {k: v for k, v in well_map.items() if int(k[1:]) not in cols}

def enforce_8x10_row_major(circles_xyz, rows=8, cols=10):
    """
    Enforce stable order:
      - split into 8 rows based on y (robust grouping)
      - within each row sort by x (left->right)
      - return exactly up to rows*cols circles in row-major order
    Also returns synthetic labels: A2..H11 (10 per row).
    """
    if not circles_xyz:
        return [], []

    # sort by y
    pts = sorted([(int(x), int(y), int(r)) for (x, y, r) in circles_xyz], key=lambda t: t[1])

    # --- robust row grouping ---
    # 1) estimate 8 row centers using quantiles of y
    ys = np.array([p[1] for p in pts], dtype=float)
    # if very few points, fallback to simple ordering
    if len(ys) < rows:
        ordered = sorted(pts, key=lambda t: (t[1], t[0]))[: rows * cols]
        labels = []
        for ri, row in enumerate("ABCDEFGH"[:rows]):
            for ci, col in enumerate(range(2, 2 + cols)):
                labels.append(f"{row}{col}")
        return ordered, labels[: len(ordered)]

    q = np.linspace(0.05, 0.95, rows)
    row_centers = np.quantile(ys, q)

    row_bins = [[] for _ in range(rows)]
    for x, y, r in pts:
        i = int(np.argmin(np.abs(row_centers - y)))
        row_bins[i].append((x, y, r))

    # If a row is empty (can happen if detection is messy), fallback to chunking
    if any(len(rb) == 0 for rb in row_bins):
        # chunk by y into 8 groups
        ordered = []
        chunk = max(1, len(pts) // rows)
        for i in range(rows):
            group = pts[i * chunk:(i + 1) * chunk] if i < rows - 1 else pts[i * chunk:]
            group = sorted(group, key=lambda t: t[0])  # sort x
            ordered.extend(group[:cols])
    else:
        ordered = []
        for rb in row_bins:
            rb = sorted(rb, key=lambda t: t[0])  # sort x
            ordered.extend(rb[:cols])

    ordered = ordered[: rows * cols]

    # A2..H11 labels (10 per row)
    labels = []
    for row in "ABCDEFGH"[:rows]:
        for col in range(2, 2 + cols):  # 2..11 for cols=10
            labels.append(f"{row}{col}")
    return ordered, labels[: len(ordered)]

# ============================================================
# MODEL INFERENCE HELPERS (copied from Lina's logic)
# ============================================================

# Oversaturation thresholds (same as Lina)
HOT_PIXEL_THRESH, CHANNEL_THRESH = 0.10, 240

# Uncertainty threshold (same as Lina)
UNCERTAINTY_CONF_THRESH = 0.70

def crop_well(img_bgr, x, y, r, size=64):
    """Crop the smallest rectangle that encloses the circle and resize to (size,size)."""
    h, w = img_bgr.shape[:2]
    x1 = max(int(x - r), 0); x2 = min(int(x + r), w)
    y1 = max(int(y - r), 0); y2 = min(int(y + r), h)
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

def is_oversaturated(crop_bgr):
    hot = (
        (crop_bgr[:, :, 0] > CHANNEL_THRESH) |
        (crop_bgr[:, :, 1] > CHANNEL_THRESH) |
        (crop_bgr[:, :, 2] > CHANNEL_THRESH)
    )
    return float(np.mean(hot)) > HOT_PIXEL_THRESH

def extract_full_features(crop_bgr):
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return np.array([
        float(lab[:, :, 0].mean()), float(lab[:, :, 1].mean()), float(lab[:, :, 2].mean()),
        float(hsv[:, :, 0].mean()), float(hsv[:, :, 1].mean()), float(hsv[:, :, 2].mean()),
        float(np.var(gray))
    ], dtype=float)

def proba_class1(model, X):
    """
    Return probability for class=1 (same robust logic as Lina).
    Your class 1 = "dead".
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # pick column corresponding to class 1 robustly
        if hasattr(model, "classes_"):
            classes = model.classes_
            if 1 in classes:
                idx = int(np.where(classes == 1)[0][0])
                return proba[:, idx]
        return proba[:, 1]
    # fallback: no proba available
    return None


class ImageViewerModel:
    """
    Model: keeps track of image file paths, current index, and per-image predictions/labels.
    predictions[path] : list of 0/1 (model output, random for now)
    labels[path]      : list of 0/1 (possibly edited by user)
    """

    def __init__(self):
        self._paths = []
        self._index = -1
        self._predictions = {}  # path -> list[int]
        self._labels = {}       # path -> list[int]
        self._img_override = {}  # path -> np.ndarray (BGR) for modified images (rotated/warped)
        self._circles = {}  # path -> list[(x,y,r)] in IMAGE pixel coords
        self._well_labels = {}  # path -> list[str] (e.g., ["A2","A3",...,"H11"])
        self._undo = {}  # path -> list of snapshots
        self._redo = {}  # path -> list of snapshots
        self._probas = {}       # path -> list[float or None] aligned with circles


    def set_images(self, paths):
        """Set new list of image paths and clear predictions/labels."""
        self._paths = paths or []
        self._index = 0 if self._paths else -1
        self._predictions.clear()
        self._labels.clear()
        self._img_override.clear()
        self._circles.clear()
        self._well_labels.clear()
        self._undo.clear()
        self._redo.clear()
        self._probas.clear()
        
    def set_predictions_for_path(self, path, labels, probas=None):
        if path is None:
            return
        self._predictions[path] = list(labels) if labels is not None else None
        self._labels[path] = list(labels) if labels is not None else None
        if probas is not None:
            self._probas[path] = list(probas)

    def get_probas_for_path(self, path):
        return self._probas.get(path)


    def push_history(self, path):
        if path is None:
            return
        snap = {
            "circles": list(self._circles.get(path, [])),
            "labels": (self._labels.get(path) or []).copy() if path in self._labels else None,
            "well_labels": list(self._well_labels.get(path, [])),
        }
        self._undo.setdefault(path, []).append(snap)
        self._redo[path] = []

    def can_undo(self, path): return bool(self._undo.get(path))
    def can_redo(self, path): return bool(self._redo.get(path))

    def undo(self, path):
        if not self.can_undo(path): return False
        cur = {
            "circles": list(self._circles.get(path, [])),
            "labels": (self._labels.get(path) or []).copy() if path in self._labels else None,
            "well_labels": list(self._well_labels.get(path, [])),
        }
        snap = self._undo[path].pop()
        self._redo.setdefault(path, []).append(cur)
        self._circles[path] = snap["circles"]
        self._well_labels[path] = snap["well_labels"]
        if snap["labels"] is None:
            self._labels.pop(path, None)
        else:
            self._labels[path] = snap["labels"]
        return True

    def redo(self, path):
        if not self.can_redo(path): return False
        cur = {
            "circles": list(self._circles.get(path, [])),
            "labels": (self._labels.get(path) or []).copy() if path in self._labels else None,
            "well_labels": list(self._well_labels.get(path, [])),
        }
        snap = self._redo[path].pop()
        self._undo.setdefault(path, []).append(cur)
        self._circles[path] = snap["circles"]
        self._well_labels[path] = snap["well_labels"]
        if snap["labels"] is None:
            self._labels.pop(path, None)
        else:
            self._labels[path] = snap["labels"]
        return True

        
    def set_well_labels(self, path, labels):
        if path is None:
            return
        self._well_labels[path] = labels or []

    def get_well_labels_for_path(self, path):
        return self._well_labels.get(path, [])


    def set_circles(self, path, circles_xyz):
        """circles_xyz: list of (x,y,r) ints in image pixel coords."""
        if path is None:
            return
        self._circles[path] = circles_xyz or []

    def get_circles_for_path(self, path):
        return self._circles.get(path, [])

    def get_circles_for_current(self):
        return self.get_circles_for_path(self.current_path())

    def has_images(self):
        return len(self._paths) > 0

    def count(self):
        return len(self._paths)

    def current_index(self):
        return self._index

    def current_path(self):
        if not self.has_images():
            return None
        return self._paths[self._index]
    
    def get_bgr(self, path):
        """Return current image pixels for a path: override (if any) else read from disk."""
        if path is None:
            return None
        if path in self._img_override:
            return self._img_override[path]
        return cv2.imread(path)

    def set_override(self, path, img_bgr):
        """Set/replace the in-memory override for this path."""
        if path is None:
            return
        if img_bgr is None:
            self._img_override.pop(path, None)
        else:
            self._img_override[path] = img_bgr

    def has_override(self, path):
        return path in self._img_override


    def current_position(self):
        if not self.has_images():
            return 0, 0
        return self._index + 1, len(self._paths)

    def can_go_next(self):
        return self.has_images() and self._index < len(self._paths) - 1

    def can_go_previous(self):
        return self.has_images() and self._index > 0

    def go_next(self):
        if self.can_go_next():
            self._index += 1
            return True
        return False

    def go_previous(self):
        if self.can_go_previous():
            self._index -= 1
            return True
        return False

    # ----- prediction / label handling -----

    def generate_random_predictions_for_all(self, num_circles=NUM_CIRCLES):
        """
        Simulate ML model: for now create random 0/1 predictions
        for ALL images and initialize labels with the same values.
        """
        self._predictions.clear()
        self._labels.clear()
        for path in self._paths:
            n = len(self._circles.get(path, [])) or num_circles
            preds = [random.randint(0, 1) for _ in range(n)]
            self._predictions[path] = preds
            self._labels[path] = preds.copy()

    def get_labels_for_current(self):
        path = self.current_path()
        if path is None:
            return None
        return self._labels.get(path)

    def has_predictions_for_current(self):
        path = self.current_path()
        if path is None:
            return False
        return path in self._predictions

    def reset_labels_for_current(self):
        """Reset current labels back to stored predictions (model output)."""
        path = self.current_path()
        if path is None:
            return
        if path in self._predictions:
            self._labels[path] = self._predictions[path].copy()

    def get_all_labels_for_path(self, path):
        return self._labels.get(path)


class ImageDisplayWidget(QWidget):
    """
    Custom widget to display an image with an overlay of clickable circles.
    - Circles have transparent interior and colored boundaries.
    - Colors are based on a 0/1 labels list (0 = red, 1 = green).
    - Clicking toggles the label for that circle.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None

        # reference to current labels list (from model), or None
        self.labels = None
        self.circles_visible = False
        
        self.edit_mode = False
        self.selected_idx = None
        self.on_selection_changed = None  # callback(idx:int|None, r:int|None)
        self._dragging = False
        self._drag_offset_img = (0.0, 0.0)  # offset between click point and center in IMAGE coords

        # callback hook set by parent; called when circles change
        self.on_circles_changed = None

        self._last_released_center = None

        self.circles_xyz = None  # list of (x,y,r) in IMAGE pixel coords


        # store last draw rect of the image to map normalized coords -> pixels
        self._last_draw_rect = None

        # precompute normalized positions for 80 circles on a simple grid
        self.circle_norm_positions = []
        cols = 10
        rows = 8
        for idx in range(NUM_CIRCLES):
            r = idx // cols
            c = idx % cols
            nx = (c + 0.5) / cols  # 0..1
            ny = (r + 0.5) / rows  # 0..1
            self.circle_norm_positions.append((nx, ny))

        # normalized radius relative to smaller axis
        self.circle_radius_norm = 0.35 * min(1.0 / cols, 1.0 / rows)

        self.setMinimumSize(400, 300)

        self.setStyleSheet(
            "background-color: #222; color: #aaa; border-radius: 5px;"
        )

    # ----- public API -----
    
    def _nearest_circle_index(self, ix, iy):
        if not self.circles_xyz:
            return None
        best_i, best_d2 = None, 1e18
        for i, (x, y, _r) in enumerate(self.circles_xyz):
            d2 = (ix - x) ** 2 + (iy - y) ** 2
            if d2 < best_d2:
                best_i, best_d2 = i, d2
        return best_i

    def set_circles_xyz(self, circles_xyz):
        self.circles_xyz = circles_xyz
        self.update()
        
    def set_edit_mode(self, enabled: bool):
        self.edit_mode = bool(enabled)
        self.update()

    def _widget_to_image_xy(self, wx, wy):
        """Map widget pixel coords -> image pixel coords using last draw rect."""
        if self._last_draw_rect is None or self.pixmap is None:
            return None
        rect = self._last_draw_rect
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        if img_w <= 0 or img_h <= 0:
            return None

        scale = min(self.width() / img_w, self.height() / img_h)
        ix = (wx - rect.x()) / scale
        iy = (wy - rect.y()) / scale
        return float(ix), float(iy)

    def _hit_test_circle(self, ix, iy):
        """Return index of circle containing (ix,iy) in IMAGE coords, else None."""
        if not self.circles_xyz:
            return None
        best = None
        best_d2 = 1e18
        for i, (x, y, r) in enumerate(self.circles_xyz):
            dx = ix - x
            dy = iy - y
            d2 = dx * dx + dy * dy
            if d2 <= (r * r) and d2 < best_d2:
                best = i
                best_d2 = d2
        return best

    def _emit_circles_changed(self):
        if callable(self.on_circles_changed):
            self.on_circles_changed(self.circles_xyz or [])


    def set_image(self, source):
        """
        source can be:
        - None
        - str path
        - np.ndarray (BGR)
        """
        if source is None:
            self.pixmap = None

        elif isinstance(source, str):
            pm = QPixmap(source)
            self.pixmap = None if pm.isNull() else pm

        else:
            # assume numpy BGR
            img_bgr = source
            if not isinstance(img_bgr, np.ndarray):
                self.pixmap = None
                self._last_draw_rect = None
                self.update()
                return

            if img_bgr.ndim == 2:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            h, w = img_rgb.shape[:2]
            img_rgb = np.ascontiguousarray(img_rgb)
            bytes_per_line = img_rgb.strides[0]
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimg)

        self._last_draw_rect = None
        self.update()


    def set_labels_ref(self, labels):
        """
        Set the labels list reference from the model.
        labels: list[int] or None
        """
        self.labels = labels
        self.circles_visible = labels is not None
        self.update()

    def clear_circles(self):
        self.labels = None
        self.circles_visible = False
        self.update()

    # ----- Qt events -----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        # background
        painter.fillRect(self.rect(), QColor(34, 34, 34))

        if not self.pixmap:
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "No images loaded")
            self._last_draw_rect = None
            return

        img_w = self.pixmap.width()
        img_h = self.pixmap.height()

        if img_w <= 0 or img_h <= 0:
            return

        scale = min(w / img_w, h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        offset_x = (w - draw_w) / 2.0
        offset_y = (h - draw_h) / 2.0

        target_rect = QRectF(offset_x, offset_y, draw_w, draw_h)
        self._last_draw_rect = target_rect

        # draw image
        painter.drawPixmap(target_rect, self.pixmap, QRectF(0, 0, img_w, img_h))

        # draw circles
        # draw circles (detected wells)
        if self.circles_xyz:
            # scale from image pixels -> widget pixels
            for idx, (x, y, r) in enumerate(self.circles_xyz):
                cx = offset_x + (x * scale)
                cy = offset_y + (y * scale)
                rr = r * scale

                # color logic:
                # - if labels exist (and have an entry), keep your green/red scheme
                # - otherwise: blue (requested)
                # selected circle should be black
                if self.edit_mode and self.selected_idx == idx:
                    color = QColor(0, 0, 0)
                else:
                    if self.labels is not None and idx < len(self.labels):
                        value = self.labels[idx]
                        # 0 = alive -> green, 1 = dead -> red, 2 = uncertain -> yellow
                        if value == 2:
                            color = QColor(255, 215, 0)  # yellow/orange
                        elif value == 1:
                            color = QColor(255, 0, 0)    # red
                        else:
                            color = QColor(0, 255, 0)    # green

                    else:
                        color = QColor(0, 0, 255)

                width = 4 if (self.edit_mode and self.selected_idx == idx) else 2
                pen = QPen(color, width)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), rr, rr)



    def mousePressEvent(self, event):
        if self._last_draw_rect is None or self.pixmap is None:
            return

        pos = event.pos()
        mapped = self._widget_to_image_xy(pos.x(), pos.y())
        if mapped is None:
            return
        ix, iy = mapped

        # EDIT MODE: select exactly ONE circle (or deselect)
        if self.edit_mode and self.circles_xyz:
            hit = self._hit_test_circle(ix, iy)

            # left click: select + start drag (only if hit)
            if event.button() == Qt.LeftButton:
                if hit is None:
                    # click on empty space: deselect
                    self.selected_idx = None
                    self._dragging = False
                    self.update()
                    return

                # select exactly one
                if self.selected_idx != hit:
                    self.selected_idx = hit
                    if callable(self.on_selection_changed):
                        _, _, rr0 = self.circles_xyz[hit]
                        self.on_selection_changed(hit, int(rr0))
                else:
                    self.selected_idx = hit  # keep as-is

                cx, cy, _ = self.circles_xyz[hit]
                self._drag_offset_img = (ix - cx, iy - cy)
                self._dragging = True
                self.update()
                return

            # right click: deselect
            if event.button() == Qt.RightButton:
                self.selected_idx = None
                self._dragging = False
                self.update()
                return

        # NOT EDITING: label toggling
        if (
            not self.edit_mode
            and self.circles_visible
            and self.labels is not None
            and self.circles_xyz
            and event.button() == Qt.LeftButton
        ):
            hit = self._hit_test_circle(ix, iy)
            if hit is not None and hit < len(self.labels):
                cur = self.labels[hit]
                if cur == 2:
                    # first manual resolve: set to "dead" (1). Next click toggles normally.
                    self.labels[hit] = 1
                else:
                    self.labels[hit] = 1 - cur
                self.update()
                return

    def mouseMoveEvent(self, event):
        if not (self.edit_mode and self._dragging and self.selected_idx is not None and self.circles_xyz):
            return
        mapped = self._widget_to_image_xy(event.pos().x(), event.pos().y())
        if mapped is None:
            return
        ix, iy = mapped

        ox, oy = self._drag_offset_img
        nx = int(round(ix - ox))
        ny = int(round(iy - oy))

        x, y, r = self.circles_xyz[self.selected_idx]
        self.circles_xyz[self.selected_idx] = (nx, ny, r)

        # IMPORTANT: don't reorder while dragging (prevents "collecting" others)
        self.update()


    def mouseReleaseEvent(self, event):
        if self.edit_mode and self._dragging:
            self._dragging = False

            # remember where the selected circle ended up (image coords)
            self._last_released_center = None
            if self.selected_idx is not None and self.circles_xyz and self.selected_idx < len(self.circles_xyz):
                x, y, _r = self.circles_xyz[self.selected_idx]
                self._last_released_center = (int(x), int(y))

            self._emit_circles_changed()
            self.update()

    # def wheelEvent(self, event):
    #     if not (self.edit_mode and self.selected_idx is not None and self.circles_xyz):
    #         return
    #     delta = event.angleDelta().y()
    #     if delta == 0:
    #         return

    #     step = 1
    #     if event.modifiers() & Qt.ShiftModifier:
    #         step = 3  # faster resize with Shift

    #     dr = step if delta > 0 else -step
    #     x, y, r = self.circles_xyz[self.selected_idx]
    #     r2 = int(np.clip(r + dr, 10, 200))
    #     self.circles_xyz[self.selected_idx] = (x, y, r2)
    #     # don't reorder immediately; just redraw
    #     self.update()
    def wheelEvent(self, event):
        # disabled: use the radius slider instead
        return


class ImageViewerWidget(QWidget):
    """
    Main viewer widget: displays the image, circles, and navigation + action buttons.
    """

    def __init__(self, model: ImageViewerModel, parent=None):
        super().__init__(parent)
        self.model = model

        # image display
        self.image_display = ImageDisplayWidget()
        
        self._last_path = None

        # right-side well size controls
        self.radius_title = QLabel("Well size")
        self.radius_title.setStyleSheet("color: #ddd; font-weight: 600;")

        self.radius_value = QLabel("Radius: -")
        self.radius_value.setStyleSheet("color: #ddd;")

        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(10, 100)
        self.radius_slider.setValue(38)
        self.radius_slider.setEnabled(False)

        # info + navigation
        self.info_label = QLabel("0 / 0")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #ddd;")

        self.prev_button = QPushButton("⟨ Previous")
        self.next_button = QPushButton("Next ⟩")

        button_style = """
            QPushButton {
                padding: 6px 12px;
                border-radius: 4px;
                background-color: #444;
                color: #eee;
                border: 1px solid #666;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #777;
                border-color: #444;
            }
        """
        self.prev_button.setStyleSheet(button_style)
        self.next_button.setStyleSheet(button_style)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.info_label)
        nav_layout.addWidget(self.next_button)

        # ----- new buttons: Rotate / Preprocess / Predict / Save / Reset -----
        self.rotate_button = QPushButton("Rotate")
        self.preprocess_button = QPushButton("Preprocess")
        self.predict_button = QPushButton("Predict")
        self.save_button = QPushButton("Save")
        self.reset_button = QPushButton("Reset")
        self.edit_wells_button = QPushButton("Edit Wells: OFF")
        self.add_well_button = QPushButton("Add Well")
        self.delete_well_button = QPushButton("Delete Selected")

        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.undo_button.setStyleSheet(button_style)
        self.redo_button.setStyleSheet(button_style)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        



        for btn in (
            self.rotate_button, self.preprocess_button,
            self.edit_wells_button, self.add_well_button, self.delete_well_button,
            self.predict_button, self.save_button, self.reset_button
        ):
            btn.setStyleSheet(button_style)



        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.rotate_button)
        actions_layout.addWidget(self.preprocess_button)
        # actions_layout.addWidget(self.edit_wells_button)
        # actions_layout.addWidget(self.add_well_button)
        actions_layout.addWidget(self.predict_button)
        # actions_layout.addWidget(self.delete_well_button)
        actions_layout.addWidget(self.save_button)
        actions_layout.addWidget(self.reset_button)

        main_layout = QVBoxLayout()
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_display, stretch=1)

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.radius_title)
        right_panel.addWidget(self.radius_value)
        right_panel.addWidget(self.radius_slider, stretch=1)
        
        right_panel.addWidget(self.undo_button)
        right_panel.addWidget(self.redo_button)


        # editing controls on the right
        right_panel.addSpacing(10)
        right_panel.addWidget(self.edit_wells_button)
        right_panel.addWidget(self.add_well_button)
        right_panel.addWidget(self.delete_well_button)

        right_panel.addStretch(1)


        image_row.addLayout(right_panel)

        main_layout.addLayout(image_row, stretch=1)
        main_layout.addLayout(nav_layout)
        main_layout.addLayout(actions_layout)
        self.setLayout(main_layout)

        # connections
        self.prev_button.clicked.connect(self.on_prev_clicked)
        self.next_button.clicked.connect(self.on_next_clicked)
        
        self.radius_slider.valueChanged.connect(self.on_radius_slider_changed)

        # initial state
        self.update_view()
        
    def on_circle_selected(self, idx, r):
        # enable slider + sync to selected radius
        self.radius_slider.blockSignals(True)
        self.radius_slider.setEnabled(True)
        self.radius_slider.setValue(int(r))
        self.radius_slider.blockSignals(False)
        self.radius_value.setText(f"Radius: {int(r)}")

    def on_radius_slider_changed(self, value):
        # only when editing and a circle is selected
        if not self.image_display.edit_mode:
            return
        idx = self.image_display.selected_idx
        if idx is None:
            return
        if not self.image_display.circles_xyz or idx >= len(self.image_display.circles_xyz):
            return

        x, y, _r = self.image_display.circles_xyz[idx]
        self.image_display.circles_xyz[idx] = (int(x), int(y), int(value))
        self.radius_value.setText(f"Radius: {int(value)}")

        # persist to model (no reordering needed; radius doesn't affect row/col order)
        path = self.model.current_path()
        self.model.set_circles(path, list(self.image_display.circles_xyz))
        self.image_display.update()


    def on_prev_clicked(self):
        if self.model.go_previous():
            self.update_view()

    def on_next_clicked(self):
        if self.model.go_next():
            self.update_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.image_display.update()

    def update_view(self):
        """
        Refresh the displayed image, labels, and buttons based on the model.
        """
        if not self.model.has_images():
            self.image_display.set_image(None)
            self.image_display.clear_circles()
            self.info_label.setText("0 / 0")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            # Predict/Save/Reset still enabled, but Save/Reset won't do anything useful
            return

        path = self.model.current_path()
        if self.model.has_override(path):
            self.image_display.set_image(self.model.get_bgr(path))
        else:
            self.image_display.set_image(path)


        current, total = self.model.current_position()
        filename = os.path.basename(path) if path else "N/A"
        self.info_label.setText(f"{current} / {total} — {filename}")

        self.prev_button.setEnabled(self.model.can_go_previous())
        self.next_button.setEnabled(self.model.can_go_next())

        # connect labels for current image (if predictions exist)
        labels = self.model.get_labels_for_current()
        self.image_display.set_labels_ref(labels)
        
        circles_xyz = self.model.get_circles_for_current()
        self.image_display.set_circles_xyz(circles_xyz)
        # selection -> slider
        self.image_display.on_selection_changed = self.on_circle_selected

        # reset slider when changing images (until a circle is clicked)
        self.radius_slider.blockSignals(True)
        self.radius_slider.setEnabled(False)
        self.radius_value.setText("Radius: -")
        self.radius_slider.blockSignals(False)

        # Only reset selection when we actually changed to a different image
        if path != getattr(self, "_last_path", None):
            self.image_display.selected_idx = None
            self.image_display._dragging = False

        self._last_path = path
        path = self.model.current_path()
        self.undo_button.setEnabled(self.model.can_undo(path))
        self.redo_button.setEnabled(self.model.can_redo(path))


        def _on_changed(new_circles):
            path = self.model.current_path()

            # preserve selection by released position (not by index)
            keep_xy = getattr(self.image_display, "_last_released_center", None)

            ordered, labels = enforce_8x10_row_major(new_circles, rows=8, cols=10)
            self.model.set_circles(path, ordered)
            self.model.set_well_labels(path, labels)

            self.image_display.circles_xyz = ordered

            # re-select the same circle after reordering
            if keep_xy is not None:
                kx, ky = keep_xy
                new_idx = self.image_display._nearest_circle_index(kx, ky)
                self.image_display.selected_idx = new_idx

                # update slider to match newly selected circle
                if new_idx is not None and new_idx < len(ordered):
                    _x, _y, rr = ordered[new_idx]
                    self.on_circle_selected(new_idx, int(rr))


        self.image_display.on_circles_changed = _on_changed


class MainWindow(QMainWindow):
    """
    Application main window: holds toolbar, viewer widget, and output dir label.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Colony Counter")
        self.resize(900, 600)

        # model
        self.model = ImageViewerModel()

        # central viewer widget
        self.viewer_widget = ImageViewerWidget(self.model)

        # connect edit-wells UI
        self.viewer_widget.edit_wells_button.clicked.connect(self.on_toggle_edit_wells)
        self.viewer_widget.add_well_button.clicked.connect(self.on_add_well_clicked)
        self.viewer_widget.delete_well_button.clicked.connect(self.on_delete_well_clicked)
        self.viewer_widget.undo_button.clicked.connect(self.on_undo)
        self.viewer_widget.redo_button.clicked.connect(self.on_redo)

        
        # preprocess
        self.viewer_widget.preprocess_button.clicked.connect(self.on_preprocess_clicked)
        
        # rotate button
        self.viewer_widget.rotate_button.clicked.connect(self.on_rotate_ccw_clicked)

        # output directory state + label
        self.output_dir = None
        self.output_dir_label = QLabel("Output directory: (none)")
        self.output_dir_label.setAlignment(Qt.AlignCenter)
        self.output_dir_label.setStyleSheet("color: #ccc; padding: 4px;")

        # 🔤 Excel filename input
        self.excel_name_label = QLabel("Excel filename (without .xlsx):")
        self.excel_name_label.setAlignment(Qt.AlignCenter)
        self.excel_name_label.setStyleSheet("color: #ccc; padding: 2px;")

        self.excel_name_edit = QLineEdit()
        self.excel_name_edit.setPlaceholderText("e.g. experiment_01")
        self.excel_name_edit.setStyleSheet("padding: 4px;")

        excel_name_layout = QHBoxLayout()
        excel_name_layout.addWidget(self.excel_name_label)
        excel_name_layout.addWidget(self.excel_name_edit)

        # central layout
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addWidget(self.viewer_widget, stretch=1)
        central_layout.addWidget(self.output_dir_label)
        central_layout.addLayout(excel_name_layout)   # ← add this line
        self.setCentralWidget(central)


        # toolbar
        self._create_toolbar()

        # connect Predict / Save / Reset buttons
        self.viewer_widget.predict_button.clicked.connect(self.on_predict_clicked)
        self.viewer_widget.save_button.clicked.connect(self.on_save_clicked)
        self.viewer_widget.reset_button.clicked.connect(self.on_reset_clicked)

        self.setStyleSheet("QMainWindow { background-color: #333; }")

    def on_undo(self):
        path = self.model.current_path()
        if self.model.undo(path):
            self.viewer_widget.update_view()

    def on_redo(self):
        path = self.model.current_path()
        if self.model.redo(path):
            self.viewer_widget.update_view()
    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.on_undo()
            return
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Y:
            self.on_redo()
            return
        super().keyPressEvent(event)

    def on_delete_well_clicked(self):
        """
        Delete the currently selected well (only works in edit mode).
        Then enforce ordering again.
        """
        if not self.model.has_images():
            return

        if not self.viewer_widget.image_display.edit_mode:
            QMessageBox.information(self, "Edit mode off", "Turn on 'Edit Wells' first.")
            return

        idx = self.viewer_widget.image_display.selected_idx
        if idx is None:
            QMessageBox.information(self, "No selection", "Click a circle to select it first.")
            return

        path = self.model.current_path()
        circles = list(self.model.get_circles_for_current() or [])
        if idx < 0 or idx >= len(circles):
            return
        
        path = self.model.current_path()

        # ⬅️ push history BEFORE deletion
        self.model.push_history(path)

        # remove it
        circles.pop(idx)

        # enforce stable 8x10 order (will keep up to 80; can be fewer after deletion)
        ordered, labels = enforce_8x10_row_major(circles, rows=8, cols=10)
        self.model.set_circles(path, ordered)
        self.model.set_well_labels(path, labels)

        # keep labels list aligned if it exists
        cur_labels = self.model.get_labels_for_current()
        if cur_labels is not None:
            n = len(ordered)
            self.model._labels[path] = (cur_labels[:n] + [0] * n)[:n]

        # clear selection after delete
        self.viewer_widget.image_display.selected_idx = None
        self.viewer_widget.image_display._dragging = False

        self.viewer_widget.update_view()


    def on_toggle_edit_wells(self):
        enabled = not self.viewer_widget.image_display.edit_mode
        self.viewer_widget.image_display.set_edit_mode(enabled)
        self.viewer_widget.edit_wells_button.setText(f"Edit Wells: {'ON' if enabled else 'OFF'}")

    def on_add_well_clicked(self):
        """
        Add a new well at the center of the view (in IMAGE coords).
        Then enforce 8x10 ordering.
        """
        if not self.model.has_images():
            return
        path = self.model.current_path()
        circles = list(self.model.get_circles_for_current() or [])

        # Need an image to pick a reasonable default center
        img = self.model.get_bgr(path)
        if img is None:
            return
        H, W = img.shape[:2]

        # default new circle in center; default radius ~ current median or 38
        if circles:
            med_r = int(np.median([c[2] for c in circles]))
        else:
            med_r = 38

        path = self.model.current_path()

        # ⬅️ push history BEFORE modification
        self.model.push_history(path)

        circles.append((W // 2, H // 2, med_r))


        # enforce stable order + store
        ordered, labels = enforce_8x10_row_major(circles, rows=8, cols=10)
        self.model.set_circles(path, ordered)
        self.model.set_well_labels(path, labels)

        # also ensure predictions/labels length matches if already predicted
        cur_labels = self.model.get_labels_for_current()
        if cur_labels is not None:
            # truncate/extend with zeros to match
            n = len(ordered)
            if len(cur_labels) != n:
                new_lab = (cur_labels[:n] + [0] * n)[:n]
                self.model._labels[path] = new_lab

        self.viewer_widget.update_view()
        # select the newly added circle (closest to image center)
        path = self.model.current_path()
        img = self.model.get_bgr(path)
        if img is not None:
            H, W = img.shape[:2]
            circles = self.model.get_circles_for_current()
            if circles:
                i = self.viewer_widget.image_display._nearest_circle_index(W // 2, H // 2)
                self.viewer_widget.image_display.selected_idx = i
                if i is not None:
                    _x, _y, rr = circles[i]
                    self.viewer_widget.on_circle_selected(i, int(rr))


    def on_rotate_ccw_clicked(self):
        """
        Rotate ONLY the currently displayed image by 90° CCW.
        - No disk writes.
        - Stores the rotated pixels as an in-memory override for that image path.
        - Preprocess will automatically use the rotated pixels (see change below).
        """
        if not self.model.has_images():
            QMessageBox.information(self, "No images", "Please open images first.")
            return

        path = self.model.current_path()
        img = self.model.get_bgr(path)
        if img is None:
            QMessageBox.warning(self, "Rotate failed", f"Could not read image:\n{path}")
            return

        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.model.set_override(path, rotated)

        self.viewer_widget.update_view()

    def on_preprocess_clicked(self):
        """
        Warps all currently loaded images IN MEMORY (no disk writes).
        Uses current pixels for each image (rotated override if present).
        Stores warped result as override for each path, so downstream uses warped images.
        """
        if not self.model.has_images():
            QMessageBox.information(self, "No images", "Please open images first.")
            return

        failures = []
        for p in list(self.model._paths):
            img = self.model.get_bgr(p)  # <-- IMPORTANT: reads rotated override if it exists
            if img is None:
                failures.append(os.path.basename(p))
                continue

            warped = extract_plate_warp(img, W=1200, H=800)
            if warped is None:
                failures.append(os.path.basename(p))
                continue

            # store warped pixels as override (no saving)
            self.model.set_override(p, warped)
            
            # detect wells on the WARPED image
            circles = remove_overlaps(detect_wells(warped), min_dist=REMOVE_OVERLAP_MINDIST)
            well_map = remove_edge_columns(label_wells_A1_H12(circles))
            
            # sanity check: expect 8x12 before dropping edges, and 8x10 after
            full_map = label_wells_A1_H12(circles)
            if len(full_map) < 96:
                print(f"[WARN] {os.path.basename(p)}: only {len(full_map)} wells labeled (expected 96).")
            well_map = remove_edge_columns(full_map)
            if len(well_map) != 80:
                print(f"[WARN] {os.path.basename(p)}: after dropping edges got {len(well_map)} wells (expected 80).")


            # enforce stable row-major order: A2..A11, B2..B11, ..., H2..H11
            ordered_labels = []
            circles_xyz = []
            for row in "ABCDEFGH":
                for col in range(2, 12):  # 2..11 (drop 1 and 12)
                    key = f"{row}{col}"
                    if key in well_map:
                        ordered_labels.append(key)
                        circles_xyz.append(well_map[key])

            self.model.set_circles(p, circles_xyz)
            self.model.set_well_labels(p, ordered_labels)


        self.viewer_widget.update_view()

        if failures:
            QMessageBox.warning(
                self,
                "Some images failed",
                "These images could not be preprocessed:\n" + "\n".join(failures[:20]) +
                ("\n..." if len(failures) > 20 else "")
            )



    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            """
            QToolBar {
                background: #2b2b2b;
                border-bottom: 1px solid #444;
            }
            QToolButton {
                color: #eee;
                padding: 4px 8px;
            }
            QToolButton:hover {
                background-color: #444;
            }
            """
        )

        open_action = QAction("Open Images…", self)
        open_action.triggered.connect(self.open_images)

        select_dir_action = QAction("Select Output Directory…", self)
        select_dir_action.triggered.connect(self.select_output_directory)

        toolbar.addAction(open_action)
        toolbar.addAction(select_dir_action)

        self.addToolBar(toolbar)

    def update_output_dir_label(self):
        if self.output_dir:
            self.output_dir_label.setText(f"Output directory: {self.output_dir}")
        else:
            self.output_dir_label.setText("Output directory: (none)")

    # ----- toolbar actions -----

    def open_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select JPG Images",
            "",
            "JPEG Images (*.jpg *.jpeg);;PNG Images (*.png);;All Files (*)",
        )

        if not files:
            return

        # filter a bit
        image_files = [
            f
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            QMessageBox.warning(self, "No images", "No valid image files were selected.")
            return

        self.model.set_images(image_files)
        self.viewer_widget.update_view()
        # circles are cleared because predictions/labels were cleared in model

    def select_output_directory(self):
        initial_dir = self.output_dir or ""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            initial_dir,
        )

        if directory:
            self.output_dir = directory
            self.update_output_dir_label()

    # ----- buttons below image viewer -----

    def on_predict_clicked(self):
        """
        Run the saved sklearn pipeline (MLP) on the CURRENT image wells.
        Produces:
        label 0 = alive (green)
        label 1 = dead  (red)
        label 2 = uncertain (yellow) if:
            - oversaturated crop OR
            - confidence < UNCERTAINTY_CONF_THRESH
        """
        if not self.model.has_images():
            QMessageBox.information(self, "No images", "Please open images first.")
            return

        path = self.model.current_path()
        if path is None:
            return

        circles = self.model.get_circles_for_path(path)
        if not circles or len(circles) == 0:
            QMessageBox.warning(self, "No wells", "No wells detected yet. Click Preprocess first.")
            return

        # load model (joblib) from same folder as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "best_model_robust_aug.joblib")
        if not os.path.exists(model_path):
            QMessageBox.warning(
                self,
                "Missing model",
                f"Could not find best_model.joblib next to the script:\n{model_path}"
            )
            return

        try:
            model = joblib.load(model_path)
        except Exception as e:
            QMessageBox.critical(self, "Model load failed", str(e))
            return

        img = self.model.get_bgr(path)
        if img is None:
            QMessageBox.critical(self, "Image read failed", "Could not read current image pixels.")
            return

        # Build crops/features in the CURRENT circle order (this respects any user reordering!)
        feats = []
        feat_indices = []
        labels_out = [2] * len(circles)     # default uncertain
        probas_out = [None] * len(circles)  # optional storage

        for i, (x, y, r) in enumerate(circles):
            crop = crop_well(img, x, y, r, size=64)
            if crop is None:
                labels_out[i] = 2
                continue

            if is_oversaturated(crop):
                labels_out[i] = 2
                continue

            feats.append(extract_full_features(crop))
            feat_indices.append(i)

        if len(feats) == 0:
            # everything uncertain
            self.model.set_predictions_for_path(path, labels_out, probas_out)
            self.viewer_widget.update_view()
            return

        X = np.vstack(feats).astype(float)
        p1 = proba_class1(model, X)

        if p1 is None:
            QMessageBox.warning(self, "No probabilities", "Model does not provide predict_proba.")
            return

        # apply Lina confidence threshold
        for j, idx in enumerate(feat_indices):
            p = float(p1[j])
            probas_out[idx] = p
            conf = max(p, 1.0 - p)

            if conf < UNCERTAINTY_CONF_THRESH:
                labels_out[idx] = 2
            else:
                labels_out[idx] = 1 if p >= 0.5 else 0

        self.model.set_predictions_for_path(path, labels_out, probas_out)
        self.viewer_widget.update_view()


    def on_reset_clicked(self):
        """
        Reset current image labels back to model predictions.
        """
        if not self.model.has_images():
            return

        if not self.model.has_predictions_for_current():
            # nothing to reset
            return

        self.model.reset_labels_for_current()
        self.viewer_widget.update_view()

    def on_save_clicked(self):
        """
        Save current image's labels into a single Excel file for all images.
        - User chooses the Excel filename in self.excel_name_edit.
        - One sheet per image (sheet name = image base name).
        - Each sheet: 8x10 grid of 0/1 and the corresponding image next to it.
        """
        if not self.model.has_images():
            QMessageBox.information(self, "No images", "Please open images first.")
            return

        if self.output_dir is None:
            QMessageBox.warning(
                self,
                "No output directory",
                "Please select an output directory first.",
            )
            return

        excel_base = self.excel_name_edit.text().strip()
        if not excel_base:
            QMessageBox.warning(
                self,
                "No Excel filename",
                "Please enter an Excel filename (without .xlsx).",
            )
            return

        # Ensure .xlsx extension
        if not excel_base.lower().endswith(".xlsx"):
            excel_filename = excel_base + ".xlsx"
        else:
            excel_filename = excel_base

        excel_path = os.path.join(self.output_dir, excel_filename)

        path = self.model.current_path()
        labels = self.model.get_labels_for_current()

        if labels is None:
            QMessageBox.information(
                self,
                "No predictions",
                "No predictions for this image yet. Click 'Predict' first.",
            )
            return

        # 8x10 layout (must match how we draw circles)
        rows, cols = 8, 10
        if len(labels) != rows * cols:
            QMessageBox.warning(
                self,
                "Label mismatch",
                f"Expected {rows*cols} labels, got {len(labels)}.",
            )
            return

        # reshape into 8x10 grid
        grid = [
            labels[r * cols:(r + 1) * cols]
            for r in range(rows)
        ]
        df = pd.DataFrame(grid)

        # Sheet name based on image base name
        base_name = os.path.basename(path)
        sheet_name, _ = os.path.splitext(base_name)

        # Determine whether to create or append to the Excel file
        file_exists = os.path.exists(excel_path)
        mode = "a" if file_exists else "w"

        # Use openpyxl engine so we can embed the image
        # if_sheet_exists='replace' requires pandas>=1.3
        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
            mode=mode,
            if_sheet_exists="replace" if file_exists else None,
        ) as writer:
            # Write the 8x10 prediction grid at A1 (row=0, col=0)
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

            # Now embed the corresponding image next to the grid
            workbook = writer.book
            sheet = workbook[sheet_name]

            try:
                img = XLImage(path)  # use original image
                scale_factor = 0.10
                img.width = int(img.width * scale_factor)
                img.height = int(img.height * scale_factor)
            except Exception as e:
                # If embedding fails, we still keep the numeric data
                print(f"Failed to embed image in Excel: {e}")
                return

            # Place the image starting at column L (12th col), row 1
            # (so it's visually "next to" the 10 columns of predictions)
            anchor_cell = "L1"
            sheet.add_image(img, anchor_cell)

            # Optionally tweak row heights for first 8 rows if you want it nicer:
            # for r in range(1, 9):
            #     sheet.row_dimensions[r].height = 20

        # No popup on success, per earlier preference

        
def main():
    app = QApplication(sys.argv)

    # if you like: adjust global font size (you already did)
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

