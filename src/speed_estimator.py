from time import time
import numpy as np


class SpeedEstimator:
    def __init__(self):
        self.speeds = {}
        self.track_ids = []
        self.track_points = {}
        self.track_previous_points = {}
        self.track_times = {}
        self.track_time_different = {}
        self.track_distance = {}
    
    def estimate_speed(self, track_id, box):
        x, y, w, h  = box
        current_point = (x, y)
        current_time = time()
        
        if track_id not in self.track_points:
            self.track_points[track_id] = current_point
            self.track_times[track_id] = current_time
        
        previous_point = self.track_points[track_id]
        previous_time = self.track_times[track_id]
        
        distance = np.sqrt((current_point[0] - previous_point[0]) ** 2 + (current_point[1] - previous_point[1]) ** 2)
        self.track_distance[track_id] = distance
        
        time_different = current_time - previous_time
        self.track_time_different[track_id] = time_different
        
        if time_different > 0:
            speed = distance / time_different
            self.speeds[track_id] = speed
        else:
            self.speeds[track_id] = 0

        self.track_points[track_id] = current_point
        self.track_times[track_id] = current_time
    
    def get_speed(self, track_id):
        return self.speeds[track_id]
    
    def get_speed_records(self):
        text = ""
        for track_id, speed in self.speeds.items():
            text += f"{track_id}: {speed} \n"
        
        return text
        
        


        