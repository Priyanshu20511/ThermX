from datetime import datetime
from multiprocessing import Pool, cpu_count
from PyQt5.QtWidgets import QProgressDialog
import os
import sys
import pandas as pd
import cv2
import pytesseract
import numpy as np
from PyQt5.QtGui import QIcon


def process_single_frame(args):
        frame_number, frame_data, rectangles_info, ocr_config = args # Unpack arguments
        cap_path = frame_data['video_path'] # Pass video path or handle cap within subprocess (more complex)

        cap = cv2.VideoCapture(cap_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release() # Release cap for this subprocess

        if not ret:
            print(f"[WARNING] Could not read frame {frame_number} in subprocess. Skipping.")
            return None # Return None if frame can't be read

        # Preprocessing (similar to your original code)
        frame_resized = cv2.resize(frame, (1280, 1024)) # Adjust if you want to crop first
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        _, frame_thresh = cv2.threshold(frame_gray, 180, 255, cv2.THRESH_BINARY)

        row_data = {"Frame": frame_number}
        for rect_data, label in rectangles_info:
            # Convert rect_data back to QRect or extract x,y,w,h
            x, y, w, h = rect_data['x'], rect_data['y'], rect_data['width'], rect_data['height']
            roi = frame_thresh[y:y+h, x:x+w]

            text = pytesseract.image_to_string(roi, config=ocr_config).strip()
            try:
                temp = float(text)
            except ValueError: # More specific exception
                temp = None
            row_data[label] = temp
        return row_data

# Set Tesseract path if installed manually
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'  # Change if needed

# New window for asking number of frame's
from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QLineEdit

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QFileDialog, QScrollArea, QHBoxLayout, QLineEdit
)
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRect, QPoint


class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.rectangles = []
        self.current_label = ""

    def set_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.setPixmap(pixmap)

    def set_label_name(self, label):
        self.current_label = label

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            # Right-click: attempt to delete a region
            clicked_point = event.pos()
            for i, (rect, label) in enumerate(self.rectangles):
                if rect.contains(clicked_point):
                    del self.rectangles[i]
                    print(f"❌ Deleted region '{label}'")
                    self.update()
                    return
        elif event.button() == Qt.LeftButton and self.current_label:
            # Left-click with label set: begin drawing
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            rect = QRect(self.start_point, self.end_point).normalized()
            self.rectangles.append((rect, self.current_label))
            print(f"📍 Region '{self.current_label}': x={rect.x()}, y={rect.y()}, width={rect.width()}, height={rect.height()}")
            self.current_label = ""
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap():
            return

        painter = QPainter(self)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)

        for rect, label in self.rectangles:
            painter.drawRect(rect)
            painter.drawText(rect.topLeft() + QPoint(5, -5), label)

        if self.drawing:
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(rect)

# New Frame window window
class FrameInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame Sampling Configuration")
        self.setFixedSize(300, 150)

        layout = QFormLayout(self)

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("e.g., 1")
        self.per_sec_input = QLineEdit()
        self.per_sec_input.setPlaceholderText("e.g., 2")

        layout.addRow("Frame:", self.frame_input)
        layout.addRow("Per (seconds):", self.per_sec_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

    def get_inputs(self):
        return self.frame_input.text().strip(), self.per_sec_input.text().strip()
    

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThermX - Annotated Thermal OCR")
        self.setGeometry(100, 50, 1300, 1150)
        self.setWindowIcon(QIcon('app_logo.ico'))

        self.video_path = None
        self.video_pixmap = None

        # Layouts
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        draw_layout = QHBoxLayout()

        # Upload Label
        self.label = QLabel("📂 Upload a thermal video")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Segoe UI", 11))

        # Upload and Run buttons
        self.upload_button = QPushButton("Upload Video")
        self.upload_button.clicked.connect(self.upload_video)

        self.process_button = QPushButton("Run OCR")
        self.process_button.clicked.connect(self.run_program)
        self.process_button.setEnabled(False)

        # Add Region input and button
        self.region_input = QLineEdit()
        self.region_input.setPlaceholderText("Enter region label (e.g., forehead)")
        self.add_region_button = QPushButton("Add Region")
        self.add_region_button.clicked.connect(self.prepare_region_input)

        self.confirm_button = QPushButton("+")
        self.confirm_button.setFixedWidth(30)
        self.confirm_button.clicked.connect(self.set_drawing_mode)

        draw_layout.addWidget(self.add_region_button)
        draw_layout.addWidget(self.region_input)
        draw_layout.addWidget(self.confirm_button)

        # Video Label area
        self.video_label = VideoLabel()
        self.video_label.setFixedSize(1280, 1024)
        self.video_label.setStyleSheet("background-color: #e0e0e0; border: 1px solid #ccc;")

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.video_label)

        # Add widgets to layouts
        control_layout.addWidget(self.upload_button)
        control_layout.addWidget(self.process_button)

        main_layout.addWidget(self.label)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(draw_layout)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.label.setText(f"✅ Selected: {file_path.split('/')[-1]}")
            self.process_button.setEnabled(True)
            self.show_first_frame(file_path)

    def show_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.label.setText("❌ Error loading video.")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.resize(frame, (1280, 1024))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_pixmap = pixmap
            self.video_label.set_pixmap(pixmap)
        else:
            self.label.setText("❌ Couldn't read frame.")

    def prepare_region_input(self):
        region = self.region_input.text().strip()
        if region:
            self.label.setText(f"📝 Draw the region for '{region}', then press +")
        else:
            self.label.setText("⚠ Please enter a region label.")

    def set_drawing_mode(self):
        label_text = self.region_input.text().strip()
        if label_text:
            self.video_label.set_label_name(label_text)
            self.label.setText(f"👆 Click and drag to draw '{label_text}'")
            self.region_input.clear()  # <-- Clear input after confirming the region label
        else:
            self.label.setText("⚠ Please enter a region label before pressing +")
    
    # Handling the OCR and excel 
    # --- Modified start_ocr method ---
    def start_ocr(self, popup_window):
        popup_window.close()

        try:
            frames_per_second = int(self.fps_input.text().strip())
            print(f"[INFO] Frames per second to process: {frames_per_second}")
        except ValueError:
            self.label.setText("⚠ Invalid input. Enter numbers only.")
            print("[ERROR] Invalid FPS input.")
            return

        if not self.video_path:
            self.label.setText("⚠ No video selected.")
            print("[ERROR] No video path found.")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.label.setText("❌ Error opening video.")
            print("[ERROR] Unable to open video.")
            return

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / actual_fps
        cap.release() # Release the main cap early

        print(f"[INFO] Actual FPS: {actual_fps}")
        print(f"[INFO] Total Frames: {total_frames}")
        print(f"[INFO] Total Seconds: {total_seconds}")

        total_target_frames = int(frames_per_second * total_seconds)
        frame_indexes_to_process = [int(i * actual_fps / frames_per_second) for i in range(total_target_frames)]

        print(f"[INFO] Total target frames to process: {len(frame_indexes_to_process)}")

        ocr_data = []

        # Prepare data for multiprocessing
        # Note: QRect objects cannot be pickled directly for multiprocessing.
        # You'll need to pass their serializable attributes (x, y, width, height).
        rectangles_for_mp = []
        for rect, label in self.video_label.rectangles:
            rectangles_for_mp.append(({'x': rect.x(), 'y': rect.y(), 'width': rect.width(), 'height': rect.height()}, label))

        # Arguments for each process
        # Pass video_path to each process so they can open their own CV2.VideoCapture object
        # This is crucial because cv2.VideoCapture objects are NOT thread/process safe.
        task_args = []
        for frame_number in frame_indexes_to_process:
            task_args.append((frame_number, {'video_path': self.video_path}, rectangles_for_mp, r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'))
        
        progress = QProgressDialog("Processing frames...", "Cancel", 0, len(frame_indexes_to_process), self)
        progress.setWindowTitle("OCR Progress")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)  # Disable closing via X
        progress.setMinimumDuration(0)
        progress.show()
        progress.resize(500.150) #the resize for the prgress window 

        cancel_flag = False

        # Use multiprocessing Pool
        # You can adjust the number of processes (e.g., cpu_count() - 1)
        with Pool(processes=cpu_count() - 1) as pool:
            # Use imap_unordered for results as they become available, not necessarily in order
            # Or map if order is strictly important (but less efficient if some tasks take longer)
            for idx, result in enumerate(pool.imap_unordered(process_single_frame, task_args)):
                if result:
                    ocr_data.append(result)

                progress.setValue(idx + 1)
                QApplication.processEvents()  # keep UI responsive

                if progress.wasCanceled():
                    print("[CANCELLED] OCR process interrupted by user.")
                    cancel_flag = True
                    break
        progress.close()

        if cancel_flag:
            self.label.setText("⚠ OCR cancelled by user.")
            return                

        if not ocr_data:
            self.label.setText("⚠ No frames processed.")
            print("[ERROR] No OCR data generated.")
            return

        # Save results
        df = pd.DataFrame(ocr_data)
        print(df.head())
        print(df.columns)
        df = df.sort_values(by="Frame")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/OCR_Thermal_{timestamp}.xlsx"
        os.makedirs("results", exist_ok=True)
        df.to_excel(filename, index=False)

        self.label.setText(f"✅ OCR complete! Excel saved to {filename}")
        print(f"[DONE] Excel saved at: {filename}")

    def run_program(self):
        if not self.video_label.rectangles:
            self.label.setText("⚠ No regions defined!")
            return

        # Popup to get frame interval and seconds to process
        popup = QWidget()
        popup.setWindowTitle("OCR Settings")
        popup.setGeometry(200, 200, 300, 120)

        layout = QVBoxLayout()
        h_layout = QHBoxLayout()

        fps_label = QLabel("frames:")
        self.fps_input = QLineEdit()
        self.fps_input.setPlaceholderText("e.g. 3")

        per_sec_label = QLabel("/sec")

        h_layout.addWidget(fps_label)
        h_layout.addWidget(self.fps_input)
        h_layout.addWidget(per_sec_label)

        ok_button = QPushButton("Start OCR")
        ok_button.clicked.connect(lambda: self.start_ocr(popup))

        layout.addLayout(h_layout)
        layout.addWidget(ok_button)
        popup.setLayout(layout)
        popup.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
