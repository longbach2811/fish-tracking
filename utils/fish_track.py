from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from .src.speed_estimator import SpeedEstimator


MODEL_PATH = "model/fish_detector.pt"
track_cfg = "cfg/custom_track.yaml"

output_folder = "video_records"
csv_folder = "csv_records"


os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

class FishTrack:
    def __init__(self, saved_video_name, saved_csv_name, ratio):
        self.saved_video_name = saved_video_name
        self.saved_csv_name = saved_csv_name

        if f"{self.saved_video_name}.avi" not in os.listdir(output_folder):
            self.output_path = os.path.join(output_folder, f"{self.saved_video_name}.avi")
        else:
            count = len([name for name in os.listdir(output_folder) if name.startswith(self.saved_video_name)])
            self.output_path = os.path.join(output_folder, f"{self.saved_video_name}_{count}.avi")
        
        if f"{self.saved_csv_name}.csv" not in os.listdir(csv_folder):
            self.csv_filename = os.path.join(csv_folder, f"{self.saved_csv_name}.csv")
            self.converted_csv_filename = os.path.join(csv_folder, f"converted_{self.saved_csv_name}.csv")
        else:
            count = len([name for name in os.listdir(csv_folder) if name.startswith(self.saved_csv_name)])
            self.csv_filename = os.path.join(csv_folder, f"{self.saved_csv_name}_{count}.csv")
            self.converted_csv_filename = os.path.join(csv_folder, f"converted_{self.saved_csv_name}_{count}.csv")
        
        self.model = YOLO(MODEL_PATH)
        self.ratio = float(ratio) if ratio != "" else 1.0
        self.speed_estimator = SpeedEstimator()
        
        self.csv_file = open(self.csv_filename, 'a')
        self.converted_csv_file = open(self.converted_csv_filename, 'a')

    def process(self, video_path, qt_frame):
        cap = cv2.VideoCapture(video_path)

        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read the video")
            return
        roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        roi_x, roi_y, roi_w, roi_h = map(int, roi)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_time = 1 / fps

        frame_width = roi_w
        frame_height = roi_h

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use 'mp4v' codec
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        track_history = defaultdict(lambda: [])

        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            if success:
                frame = frame[int(roi_y):int(roi_y + roi_h),  
                            int(roi_x):int(roi_x + roi_w)] 
                results = self.model.track(frame, persist=True, stream=True, tracker=track_cfg)
                annotated_frame = frame.copy()
                frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
                video_time = frame_index / fps
                for r in results:
                    
                    boxes = r.boxes.xywh.cpu()
                    track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else []
                    scores = r.boxes.conf.float()
                
                    for index, (box, track_id, score) in enumerate(zip(boxes, track_ids, scores)):
                        if score < 0.5:
                            continue
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        self.speed_estimator.estimate_speed(track_id, box)
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.rectangle(annotated_frame, (int(x-w/2), int(y-h/2)), (int(x + w/2), int(y + h/2)), color=(0, 0, 255), thickness=2)
                        cv2.putText(annotated_frame, str(track_id), (int(x-w/2), int(y-h/2) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            
                        text = f"{track_id}: {self.speed_estimator.speeds[track_id]:.4f}"
                        y = 20 + index * 20  # Adjust Y position for each line
                        cv2.putText(annotated_frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        trkd_speed = self.speed_estimator.speeds[track_id]
                        trkd_distance = self.speed_estimator.track_distance[track_id]
                        trkd_point = self.speed_estimator.track_points[track_id]
                        
                        self.csv_file.write(
                            f"{video_time}, {track_id}, {trkd_speed}, {trkd_distance}, {trkd_point[0]}, {trkd_point[1]}\n"
                        )
                        self.converted_csv_file.write(
                            f"{video_time}, {track_id}, {trkd_speed * self.ratio}, {trkd_distance * self.ratio}, {trkd_point[0] * self.ratio}, {trkd_point[1] * self.ratio}\n"
                        )
                        
                out.write(annotated_frame)
                
                height, width, channels = annotated_frame.shape
                bytes_per_line = channels * width
                q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                qt_pixmap = QPixmap.fromImage(q_img).scaled(qt_frame.width(), qt_frame.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                qt_frame.setPixmap(qt_pixmap)
                qt_frame.repaint()
                
                elapsed_time = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed_time)  # Ensure non-negative sleep time
                time.sleep(sleep_time)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break      
        cap.release()
        out.release()

