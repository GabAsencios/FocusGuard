import time
import torch
from ultralytics import YOLO


class FocusGuardWebcam:
    """
    Webcam component for distraction detection with jitter protection.

    This class uses YOLOv8 to detect people and cell phones, applying
    temporal filtering and a grace period to handle weak or flickering
    detections common in varied lighting.

    Args:
        model_path (str): Path to the weights file (e.g., 'models/yolov8n.pt').
        conf_threshold (float): Confidence to accept a detection (0.0 to 1.0).
        event_threshold (int): Seconds a distraction must persist to trigger.
        grace_period (float): Seconds to wait before resetting the timer.

    Returns:
        FocusGuardWebcam: An initialized instance of the detector.

    Example:
        detector = FocusGuardWebcam('models/yolov8n.pt')
    """

    def __init__(self, model_path, conf_threshold=0.4, event_threshold=3, grace_period=0.5):
        # Determine device for RTX 2070 utilization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)

        # Hyperparameters for threshold tuning
        self.conf = conf_threshold
        self.event_threshold = event_threshold
        self.grace_period = grace_period

        # CRITICAL: Initialize the attributes that caused the error
        self.last_seen_times = {"cell phone": 0, "person": 0}
        self.start_times = {"cell phone": None, "absence": None}

        # Targeted COCO classes
        self.target_classes = [0, 67]

        print(f"FocusGuard initialized on: {self.device.upper()}")

    def detect_and_filter(self, frame):
        """
        Runs inference and applies temporal logic to confirm distractions.

        Args:
            frame (ndarray): The raw image frame from the webcam.

        Returns:
            tuple: (list of confirmed events, ndarray annotated_frame)
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf,
            classes=self.target_classes
        )[0]

        annotated_frame = results.plot()
        detected_names = [results.names[int(box.cls)] for box in results.boxes]
        current_time = time.time()
        confirmed_events = []

        # Phone Detection Logic (with Grace Period)
        if "cell phone" in detected_names:
            self.last_seen_times["cell phone"] = current_time
            if self.start_times["cell phone"] is None:
                self.start_times["cell phone"] = current_time
            elif current_time - self.start_times["cell phone"] >= self.event_threshold:
                confirmed_events.append("Phone Distraction")
        else:
            if self.start_times["cell phone"] and (
                    current_time - self.last_seen_times["cell phone"] > self.grace_period):
                self.start_times["cell phone"] = None

        # User Absence Logic
        if "person" not in detected_names:
            if self.start_times["absence"] is None:
                self.start_times["absence"] = current_time
            elif current_time - self.start_times["absence"] >= self.event_threshold:
                confirmed_events.append("User Absent")
        else:
            self.start_times["absence"] = None

        return confirmed_events, annotated_frame