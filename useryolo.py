import cv2
import os
import time
from ultralytics import YOLO
from playsound import playsound 
import threading
import tkinter as tk
from tkinter import messagebox

# load model and video
yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture(0)

# dim screen brightness (will have to be changed for windows)
def dim_screen():
    os.system("brightness 0.05")
def reset_brightness():
    os.system("brightness 1.0")

# Load the alert sound
alert_sound = "warning.mp3"

def start_detection():
    global running
    running = True
    start_button.config(state=tk.DISABLED)  # Disable start button to prevent multiple clicks
    stop_button.config(state=tk.NORMAL)  # Enable stop button
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.start()

# Function to stop detection
def stop_detection():
    global running
    running = False
    last_alert_time = 0  # Prevents continuous dimming
    dimmed = False  # Tracks if the screen is already dimmed
    reset_brightness()  # Restore brightness when stopping
    messagebox.showinfo("Info", "Detection Stopped")
    start_button.config(state=tk.NORMAL)  # Enable start button
    stop_button.config(state=tk.DISABLED)  # Disable stop button

def run_detection():
    global dimmed
    dimmed = False

    while running:
        ret, frame = videoCap.read()
        if not ret:
            continue

        results = yolo.track(frame, stream=True)
        person_count = 0  # Count how many people are detected

        for result in results:
            for box in result.boxes:
                if box.conf[0] > 0.4:  # Confidence value
                    cls = int(box.cls[0])
                    if cls == 0:  # Class 0 = Person
                        person_count += 1

        # If more than one person is detected, dim the screen
        if person_count > 1:
            if not dimmed:
                print("More than one person detected! Decreasing brightness")
                playsound(alert_sound)
                last_alert_time = time.time()
                dim_screen()
                dimmed = True
        else:
            if dimmed:
                print("Only one person left. Restoring brightness.")
                reset_brightness()
                dimmed = False
    window.update_idletasks()
    window.update()


window = tk.Tk()
window.title("Shoulder Surfing Detection")
start_button = tk.Button(window, text="Start Detection", command=start_detection)
start_button.pack()
stop_button = tk.Button(window, text="Stop Detection", command=stop_detection, state=tk.DISABLED)
stop_button.pack()
window.protocol("WM_DELETE_WINDOW", stop_detection)  # Handle window close
window.mainloop()

videoCap.release()
cv2.destroyAllWindows()