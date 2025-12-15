import sys
import os
import random

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
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF

import pandas as pd
from openpyxl.drawing.image import Image as XLImage


NUM_CIRCLES = 80  # number of petri dishes (circles)


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

    def set_images(self, paths):
        """Set new list of image paths and clear predictions/labels."""
        self._paths = paths or []
        self._index = 0 if self._paths else -1
        self._predictions.clear()
        self._labels.clear()

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
            preds = [random.randint(0, 1) for _ in range(num_circles)]
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

    def set_image(self, path):
        if path is None:
            self.pixmap = None
        else:
            pm = QPixmap(path)
            if pm.isNull():
                self.pixmap = None
            else:
                self.pixmap = pm
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
        if self.circles_visible and self.labels is not None:
            radius = self.circle_radius_norm * min(draw_w, draw_h)
            for idx, (nx, ny) in enumerate(self.circle_norm_positions):
                if idx >= len(self.labels):
                    break
                cx = offset_x + nx * draw_w
                cy = offset_y + ny * draw_h

                value = self.labels[idx]
                color = QColor(0, 255, 0) if value == 1 else QColor(255, 0, 0)

                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), radius, radius)

    def mousePressEvent(self, event):
        if (
            not self.circles_visible
            or self.labels is None
            or self._last_draw_rect is None
        ):
            return

        pos = event.pos()
        x = pos.x()
        y = pos.y()

        rect = self._last_draw_rect
        draw_w = rect.width()
        draw_h = rect.height()
        offset_x = rect.x()
        offset_y = rect.y()
        radius = self.circle_radius_norm * min(draw_w, draw_h)
        radius_sq = radius * radius

        # find circle under click
        for idx, (nx, ny) in enumerate(self.circle_norm_positions):
            if idx >= len(self.labels):
                break
            cx = offset_x + nx * draw_w
            cy = offset_y + ny * draw_h

            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= radius_sq:
                # toggle label 0 <-> 1
                self.labels[idx] = 1 - self.labels[idx]
                self.update()
                break

        # don't call super().mousePressEvent(event) to keep clicks local


class ImageViewerWidget(QWidget):
    """
    Main viewer widget: displays the image, circles, and navigation + action buttons.
    """

    def __init__(self, model: ImageViewerModel, parent=None):
        super().__init__(parent)
        self.model = model

        # image display
        self.image_display = ImageDisplayWidget()

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

        # ----- new buttons: Predict / Save / Reset -----
        self.predict_button = QPushButton("Predict")
        self.save_button = QPushButton("Save")
        self.reset_button = QPushButton("Reset")

        for btn in (self.predict_button, self.save_button, self.reset_button):
            btn.setStyleSheet(button_style)

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.predict_button)
        actions_layout.addWidget(self.save_button)
        actions_layout.addWidget(self.reset_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_display, stretch=1)
        main_layout.addLayout(nav_layout)
        main_layout.addLayout(actions_layout)
        self.setLayout(main_layout)

        # connections
        self.prev_button.clicked.connect(self.on_prev_clicked)
        self.next_button.clicked.connect(self.on_next_clicked)

        # initial state
        self.update_view()

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
        self.image_display.set_image(path)

        current, total = self.model.current_position()
        filename = os.path.basename(path) if path else "N/A"
        self.info_label.setText(f"{current} / {total} — {filename}")

        self.prev_button.setEnabled(self.model.can_go_previous())
        self.next_button.setEnabled(self.model.can_go_next())

        # connect labels for current image (if predictions exist)
        labels = self.model.get_labels_for_current()
        self.image_display.set_labels_ref(labels)


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
        Simulate model predictions: generate random labels for ALL images.
        Then update current view so circles appear.
        """
        if not self.model.has_images():
            QMessageBox.information(self, "No images", "Please open images first.")
            return

        self.model.generate_random_predictions_for_all(NUM_CIRCLES)
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

