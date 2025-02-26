from PyQt5 import QtCore, QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal
from utils.fish_track import FishTrack

class ProcessingThread(QThread):
    finished = pyqtSignal()

    def __init__(self, fish_track, video_path, video_frame):
        super().__init__()
        self.fish_track = fish_track
        self.video_path = video_path
        self.video_frame = video_frame

    def run(self):
        self.fish_track.process(self.video_path, self.video_frame)
        self.finished.emit()

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        loadUi("fish_tracking_ui.ui", self)
        
        self.btn_process.clicked.connect(self.process_handler)
    
    def process_handler(self):
        saved_video_name = self.txt_video_name.text()
        saved_csv_name = self.txt_csv_name.text()
        ratio = self.txt_ratio.text()
        
        self.fish_track = FishTrack(saved_video_name, saved_csv_name, ratio)

        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select video file", "", "Video Files (*.mp4 *.avi, *mov)")
        
        self.processing_thread = ProcessingThread(self.fish_track, video_path, self.video_frame)
        self.processing_thread.start()
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = MainApp()
    ui.show()
    app.exec_()