import cv2
import os
import time
import threading
import tkinter as tk
import customtkinter
from ultralytics import YOLO
from PIL import Image, ImageTk
from gaze_tracking import GazeTracking

# Load YOLO model for person detection
yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture(0)
print("Camera ON")

# Initialize gaze tracking
gaze = GazeTracking()

alert_sound = "warning2.mp3"

def dim_screen():
    try:
        import screen_brightness_control as sbc
        sbc.set_brightness(10)
    except ImportError:
        os.system("brightness 0.03")

def reset_brightness():
    try:
        import screen_brightness_control as sbc
        sbc.set_brightness(100)
    except ImportError:
        os.system("brightness 1.0")

# Create the GUI
class Window(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Shoulder Surfing & Gaze Detector") 
        self.geometry("1430x768")

        # Sidebar with buttons
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="ns")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="System Settings", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_start = customtkinter.CTkButton(self.sidebar_frame, text="Start Detection", command=self.start_detection)
        self.sidebar_button_start.grid(row=1, column=0, padx=20, pady=10)
        
        self.sidebar_button_stop = customtkinter.CTkButton(self.sidebar_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.sidebar_button_stop.grid(row=2, column=0, padx=20, pady=10)
        
        self.sidebar_button_stats = customtkinter.CTkButton(self.sidebar_frame, text="Stats", command=self.stats_button)
        self.sidebar_button_stats.grid(row=3, column=0, padx=20, pady=10)

        # Video display area
        self.video_label = tk.Label(self)
        self.video_label.grid(row=0, column=1, padx=10, pady=10)

    def stats_button(self):
        print("Stats button clicked")
    
    def start_detection(self):
        global running
        running = True
        self.sidebar_button_start.configure(state=tk.DISABLED)
        self.sidebar_button_stop.configure(state=tk.NORMAL)
        detection_thread = threading.Thread(target=self.run_detection)
        detection_thread.start()

    def stop_detection(self):
        global running
        running = False
        reset_brightness()
        self.sidebar_button_start.configure(state=tk.NORMAL)
        self.sidebar_button_stop.configure(state=tk.DISABLED)

    def run_detection(self):
        global dimmed
        dimmed = False
        
        while running:
            ret, frame = videoCap.read()
            if not ret:
                continue
            
            results = list(yolo.track(frame, stream=True))
            person_count = 0
            gaze_detected = False

            for result in results:
                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        cls = int(box.cls[0])
                        if cls == 0:
                            person_count += 1
            
            gaze.refresh(frame)
            
            if person_count > 1 and (gaze.is_right() or gaze.is_left() or gaze.is_center()):
                gaze_detected = True

            if gaze_detected:
                if not dimmed:
                    print("Potential shoulder surfer detected looking at screen! Dimming screen")
                    dim_screen()
                    dimmed = True
            else:
                if dimmed:
                    print("No unauthorized gazes detected. Restoring brightness.")
                    reset_brightness()
                    dimmed = False
            
            self.show_frame(gaze.annotated_frame())

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

if __name__ == "__main__":
    window = Window()
    window.mainloop()
    
    videoCap.release()
    print("Camera OFF")
    cv2.destroyAllWindows()