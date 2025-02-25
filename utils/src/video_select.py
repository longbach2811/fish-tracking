import tkinter as tk
from tkinter import filedialog

def select_video_file():
    # Initialize the Tkinter root window (it won't be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select video file
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )

    if file_path:
        print(f"Selected video file: {file_path}")
    else:
        print("No file selected.")
    
    return file_path