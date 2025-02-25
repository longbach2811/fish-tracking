from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os

from .src.speed_estimator import SpeedEstimator


MODEL_PATH = "../model/fish_detector.pt"
track_cfg = "../cfg/custom_track.yaml"

output_folder = "../video_records"
csv_folder = "../csv_records"


os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

class FishTrack:
    def __init__(self, saved_video_name, saved_csv_name, ratio):
        self.saved_video_name = saved_video_name
        self.saved_csv_name = saved_csv_name

        if self.saved_video_name not in os.listdir(output_folder):
            self.output_path = f"{output_folder}/{self.saved_video_name}.avi"
        else:
            self.output_path = f"{output_folder}/{self.saved_video_name}_{len([name for name in os.listdir(output_folder) if name == saved_video_name])}.avi"
        
        if self.saved_csv_name not in os.listdir(csv_folder):
            self.csv_filename = f"{csv_folder}/{self.saved_csv_name}.csv"
        else:
            self.csv_filename = f"{csv_folder}/{self.saved_csv_name}_{len([name for name in os.listdir(csv_folder) if name == saved_csv_name])}.csv"

        self.model = YOLO(MODEL_PATH)

        self.ratio = float(ratio) if ratio != "" else 1.0
        self.speed_estimator = SpeedEstimator()
        self.csv_file = open(self.csv_filename, 'a')

    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)

        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read the video")
            return
        roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        roi_x, roi_y, roi_w, roi_h = map(int, roi)

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_width = roi_w
        frame_height = roi_h

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use 'mp4v' codec
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        track_history = defaultdict(lambda: [])

        while cap.isOpened():
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

                        self.csv_file.write(
                            f"{video_time}, {track_id}, {self.speed_estimator.speeds[track_id] * self.ratio}, {self.speed_estimator.track_distance[track_id] * self.ratio}\n"
                        )
                out.write(annotated_frame)

                cv2.imshow("Tracking results", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
