import csv
import os

def save_to_csv(video_time, track_ids, speed_estimator):
    record_folder = 'csv_records'
    if 'speed_record.csv' not in  os.listdir(record_folder):
        filename = f"{record_folder}/speed_record.csv"
    else:
        filename = f"{record_folder}/speed_record_{len(os.listdir(record_folder))}.csv"
    
    with open(filename, 'a') as csv_file:
        for track_id in track_ids:
            csv_file.write(
                f"{video_time}, {speed_estimator.speeds[track_id]}, {speed_estimator.track_distance[track_id]} \n"
                )