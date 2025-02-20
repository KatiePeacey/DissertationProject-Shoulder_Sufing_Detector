import cv2
import os
import time
from ultralytics import YOLO
from playsound import playsound 
import threading
import tkinter as tk
from functools import partial
from PIL import Image, ImageTk

# load model and video
yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture(0)
print("Camera ON")

# dim screen brightness (will have to be changed for windows)
def dim_screen():
    os.system("brightness 0.03")
def reset_brightness():
    os.system("brightness 1.0")

# Load the alert sound
alert_sound = "warning.mp3"

def start_detection():
    global running
    running = True
    start_button.config(state=tk.DISABLED)  # Disable start button to prevent multiple clicks
    stop_button.config(state=tk.NORMAL)  # Enable stop button
    detection = threading.Thread(target=run_detection)
    detection.start()

def stop_detection():
    global running
    running = False
    reset_brightness()
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
                print("Potential shoulder surfer! Decreasing brightness")
                playsound(alert_sound)
                last_alert_time = time.time()
                dim_screen()
                dimmed = True
        else:
            if dimmed:
                print("Only one person left. Restoring brightness.")
                reset_brightness()
                dimmed = False

        # just for demo purposes  
        show_frame(frame)

    window.update_idletasks()
    window.update()


# Display video in Tkinter - demo purposes
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk  # Keep reference
    video_label.config(image=imgtk)

# Create the GUI
window = tk.Tk()
window.title("Shoulder Surfing Detector")
window.geometry('1430x768') 
# window.geometry('300x100') 

video_label = tk.Label(window)
video_label.pack()

button_frame = tk.Frame(window)
button_frame.pack(side=tk.BOTTOM)

start_button = tk.Button(button_frame, text="Start Detection", command=start_detection, width=20, height=2)
#start_button.pack()
start_button.grid(row=0, column=0)
stop_button = tk.Button(button_frame, text="Stop Detection", command=stop_detection, state=tk.DISABLED, width=20, height=2)
#stop_button.pack()
stop_button.grid(row=0, column=1)

window.mainloop()
videoCap.release()
print("Camera OFF")
cv2.destroyAllWindows()