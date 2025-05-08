############################################################################################################################################################
#                   Advanced Sky Fighter Jet and Drone Detector with Enhanced Features  
#                   This script implements a real-time detection and tracking system for fighter jets and drones using YOLOv8.    
#                   It includes a radar-like display, target locking, and firing simulation with sound alerts.
#                   The system is designed to be user-friendly and efficient, providing a comprehensive solution for aerial surveillance.
#                   Author: Shiboshree Roy
#                   Date: 2025-05-07
#                   License: MIT
############################################################################################################################################################



##########################################################################################################################################################
#                    Import necessary libraries
##########################################################################################################################################################

import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
import logging
from ultralytics import YOLO
from sort.sort import Sort  # SORT tracker
import winsound  # For alert sound (Windows-specific)

###########################################################################################################################
# Suppress warnings from OpenCV
# Setup logging
###########################################################################################################################


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class SkyFeatureDetector:
    def __init__(self, output_dir: str = "output"):
        """Initialize the detector with camera, YOLO model, and tracker."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
#################################################################################################################
#                    Initialize camera
#################################################################################################################


        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Failed to open camera.")
        logging.info("Camera initialized.")
##################################################################################################################
#               Load YOLO model (assumes yolov8n.pt is downloaded)
##################################################################################################################


        self.model = YOLO('yolov8n.pt')  # [if you wanna change  your  won custom  model  you can  eassily  Replace with custom model if trained]

        logging.info("YOLOv8 model loaded.")

###################################################################################################################
#           Initialize SORT tracker
###################################################################################################################


        self.tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        self.targets = []

####################################################################################################################
#        For target locking
####################################################################################################################

        self.locked_target = None 

    def read_frame(self):
        """Read a frame from the camera."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera.")
                return None
            return frame
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            return None

    def detect_objects(self, frame):
        """Detect objects using YOLOv8."""
        try:
            results = self.model(frame)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
######################################################################################################################
#                        Map COCO classes: 4 - airplane (Jet), 14 - bird (Drone proxy)
######################################################################################################################


                    if cls == 4:  #   [Airplane -> Jet] 
                        detections.append([x1, y1, x2, y2, conf, cls])

                    elif cls == 14:  #  [Bird -> Drone]


                        detections.append([x1, y1, x2, y2, conf, cls])
            return np.array(detections)
        except Exception as e:
            logging.error(f"Error detecting objects: {e}")
            return np.array([])

    def track_objects(self, detections):


###########################################################################################################
#                           Track detected objects across frames. 
###########################################################################################################


        try:
            if len(detections) > 0:
                self.targets = self.tracker.update(detections[:, :5])
            else:
                self.targets = self.tracker.update(np.empty((0, 5)))
            return self.targets
        except Exception as e:
            logging.error(f"Error tracking objects: {e}")
            return np.array([])

    def classify_target(self, cls):

############################################################################################################
#               Classify target based on YOLO class ID.
############################################################################################################


        if cls == 4:
            return "Jet"
        elif cls == 14:
            return "Drone"
        return "Unknown"

    def process_frame(self, frame):
  
############################################################################################################
#             Process a frame: detect, track, classify, and simulate firing.  
############################################################################################################

        try:
            
            detections = self.detect_objects(frame) # [Detect objects]
            if len(detections) == 0:
                self.locked_target = None  # [Reset lock if no targets]
                return frame, []  # [No firing, no targets]

#############################################################################################################
#                                   Track objects
#############################################################################################################

            tracked_objects = self.track_objects(detections)
            overlay = frame.copy()
            targets = []


#############################################################################################################
#                                   Associate detections with tracks (simplified)
#############################################################################################################

            for track in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, track)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

#############################################################################################################
#                               Find corresponding detection class
#############################################################################################################


                cls = None
                for det in detections:
                    det_x1, det_y1, det_x2, det_y2, _, det_cls = det
                    if (abs(det_x1 - x1) < 10 and abs(det_y1 - y1) < 10):
                        cls = int(det_cls)
                        break
                if cls is None:
                    continue  # [Skip if no matching detection]
                target_type = self.classify_target(cls)
                targets.append((center, target_type, track_id))

#############################################################################################################
#                   Draw bounding box and label
#############################################################################################################

                color = (0, 255, 0) if track_id != self.locked_target else (255, 0, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"{target_type} ID:{track_id}{' (Locked)' if track_id == self.locked_target else ''}"
                cv2.putText(overlay, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#############################################################################################################
#               Target locking: Lock onto first detected target
#############################################################################################################


                if self.locked_target is None and targets:
                    self.locked_target = track_id
                    logging.info(f"Locked onto {target_type} ID:{track_id}")
#############################################################################################################
#                   Simulate firing only at locked target
#############################################################################################################


                if track_id == self.locked_target:
                    fire_origin = (frame.shape[1] // 2, frame.shape[0])


#############################################################################################################
#                   Enhanced firing simulation: Laser effect
#############################################################################################################


                    for i in range(5):
                        alpha = i / 5
                        pos = (int(fire_origin[0] * (1 - alpha) + center[0] * alpha),
                               int(fire_origin[1] * (1 - alpha) + center[1] * alpha))
                        cv2.circle(overlay, pos, 5 - i, (0, 0, 255), -1)
                    cv2.line(overlay, fire_origin, center, (0, 0, 255), 2)
                    logging.info(f"Fired at {target_type} ID:{track_id} at {center}")


#############################################################################################################
#                   Save screenshot
#############################################################################################################


                    cv2.imwrite(os.path.join(self.output_dir, f"target_{track_id}_{int(cv2.getTickCount())}.jpg"), overlay)

#############################################################################################################
#               Alert if new targets are detected
#############################################################################################################


            if len(targets) > 0:
                winsound.Beep(1000, 200)  # [Alert sound]

            return overlay, targets
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame, []

    def release(self):
        """Release camera resources."""
        try:
            self.cap.release()
            logging.info("Camera released.")
        except Exception as e:
            logging.error(f"Error releasing camera: {e}")

class App:
    def __init__(self, window, window_title, detector):
#############################################################################################################
#                   Initialize the Tkinter GUI.
#############################################################################################################


        self.window = window
        self.window.title(window_title)
        self.detector = detector
        self.frame_width = 640
        self.frame_height = 480

##############################################################################################################
#                    Layout frames
##############################################################################################################


        self.video_frame = tk.Frame(window)
        self.video_frame.pack(side=tk.LEFT, padx=5, pady=5)

        self.radar_frame = tk.Frame(window)
        self.radar_frame.pack(side=tk.RIGHT, padx=5, pady=5)
#############################################################################################################
#               Video canvas
#############################################################################################################


        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.video_canvas.pack()


#############################################################################################################
#               Radar canvas
#############################################################################################################
        self.radar_canvas = tk.Canvas(self.radar_frame, width=200, height=200, bg='black')
        self.radar_canvas.pack()
        self.radar_canvas.create_oval(20, 20, 180, 180, outline='green')
        self.radar_canvas.create_line(100, 20, 100, 180, fill='green')
        self.radar_canvas.create_line(20, 100, 180, 100, fill='green')


#############################################################################################################
#       Status label
#############################################################################################################


        self.status_label = tk.Label(self.radar_frame, text="Jets: 0, Drones: 0")
        self.status_label.pack(pady=5)

        # Quit button
        self.btn_quit = tk.Button(window, text="Quit", command=self.quit)
        self.btn_quit.pack(side=tk.BOTTOM, pady=5)

        self.delay = 10  # Update every 10ms
        self.update()
        self.window.mainloop()

    def update(self):
        """Update GUI with processed frame and radar."""
        frame = self.detector.read_frame()
        if frame is not None:
            # Set frame dimensions
            if self.frame_width == 640 and self.frame_height == 480:
                self.frame_width = frame.shape[1]
                self.frame_height = frame.shape[0]
                self.video_canvas.config(width=self.frame_width, height=self.frame_height)

            # Process frame
            processed_frame, targets = self.detector.process_frame(frame)
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Update radar
            self.radar_canvas.delete('targets')
            radar_center_x, radar_center_y = 100, 100
            r = 80
            fov = 60
            for (x, y), target_type, track_id in targets:
                theta = (x / self.frame_width - 0.5) * fov
                theta_rad = np.deg2rad(theta)
                radar_x = radar_center_x + r * np.sin(theta_rad)
                radar_y = radar_center_y - r * np.cos(theta_rad)
                color = 'red' if target_type == "Jet" else 'blue'
                self.radar_canvas.create_oval(radar_x-3, radar_y-3, radar_x+3, radar_y+3, 
                                              fill=color, tags='targets')

            # Update status
            jets = sum(1 for _, t, _ in targets if t == "Jet")
            drones = sum(1 for _, t, _ in targets if t == "Drone")
            self.status_label.config(text=f"Jets: {jets}, Drones: {drones}")

        self.window.after(self.delay, self.update)

    def quit(self):
        """Clean up and exit."""
        try:
            self.detector.release()
            self.window.quit()
            logging.info("Application closed.")
        except Exception as e:
            logging.error(f"Error during quit: {e}")

def main():
    """Start the application."""
    try:
        detector = SkyFeatureDetector()
        App(tk.Tk(), "Advanced Sky Fighter Jet and Drone Detector", detector)
    except Exception as e:
        logging.error(f"Application failed: {e}")

if __name__ == "__main__":
    main()