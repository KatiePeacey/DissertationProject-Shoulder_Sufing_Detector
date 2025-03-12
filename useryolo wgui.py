import cv2
import os
from ultralytics import YOLO
from playsound import playsound 
import threading
import tkinter as tk
import customtkinter
from PIL import Image, ImageTk
import customtkinter as ctk

# load model and video
yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture(0)
print("Camera ON")

# Load the alert sound
alert_sound = "warning2.mp3"

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue") 

# Create the GUI
class Window(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Shoulder Surfing Detector") 
        self.geometry(f"{1100}x{580}")

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="System Settings", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_start = customtkinter.CTkButton(self.sidebar_frame, text="Start Detection", command=self.start_detection)
        self.sidebar_button_start.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_stop = customtkinter.CTkButton(self.sidebar_frame, text="Stop Detection", command=self.stop_detection)
        self.sidebar_button_stop.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_stats = customtkinter.CTkButton(self.sidebar_frame, text="Stats", command=self.stats_button)
        self.sidebar_button_stats.grid(row=3, column=0, padx=20, pady=10)

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=20)
        self.show_frame()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def stats_button(self):
        print("stats button click")

    # dim screen brightness (will have to be changed for windows)
    def dim_screen(self):
        os.system("brightness 0.00")
    def reset_brightness(self):
        os.system("brightness 1.0")

    def show_frame(self):
        """Capture frame from webcam and display it in the label."""
        ret, frame = self.cap.read()  # ✅ Get video frame
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = Image.fromarray(frame)  # Convert to PIL image
            imgtk = ImageTk.PhotoImage(image=img)  # Convert to Tkinter-compatible image

            # ✅ Use self.video_label
            self.video_label.imgtk = imgtk  # Prevent garbage collection
            self.video_label.configure(image=imgtk)  # Update label with new image

        # ✅ Use self.show_frame in after()
        self.video_label.after(10, self.show_frame)  # Call function every 10ms



    def start_detection(self):
        global running
        running = True
        self.sidebar_button_start.configure(state=tk.DISABLED)  # Disable start button to prevent multiple clicks
        self.sidebar_button_stop.configure(state=tk.NORMAL)  # Enable stop button
        detection = threading.Thread(target=self.run_detection)
        detection.start()

    def stop_detection(self):
        global running
        running = False
        self.reset_brightness()
        self.sidebar_button_start.configure(state=tk.NORMAL)  # Enable start button
        self.sidebar_button_stop.configure(state=tk.DISABLED)  # Disable stop button

    def run_detection(self):
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
                    # playsound(alert_sound)
                    self.dim_screen()
                    dimmed = True
            else:
                if dimmed:
                    print("Only one person left. Restoring brightness.")
                    self.reset_brightness()
                    dimmed = False

            # Start video stream
            show_frame(self)

        window.update_idletasks()
        window.update()

if __name__ == "__main__":
    window = Window()
    window.mainloop()

videoCap.release()
print("Camera OFF")
cv2.destroyAllWindows()