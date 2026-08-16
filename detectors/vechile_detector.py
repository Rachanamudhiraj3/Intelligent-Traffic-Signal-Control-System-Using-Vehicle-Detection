import time
import cv2
from ultralytics import YOLO

from config import (
    VEHICLE_MODEL_PATH,
    VEHICLE_CLASSES,
    VEHICLE_CONFIDENCE,
    VEHICLE_BOX_COLOR
)


class VehicleDetector:
    """
    Detect vehicles using YOLO.
    """

    def __init__(self):

        print("Loading Vehicle Detection Model...")

        self.model = YOLO(str(VEHICLE_MODEL_PATH))

        print("Vehicle Model Loaded Successfully!\n")

    def detect(self, image):
        """
        Detect vehicles in the given image.

        Returns
        -------
        vehicles : list
        vehicle_count : int
        inference_time : float
        """

        start_time = time.perf_counter()

        results = self.model(
            image,
            verbose=False
        )[0]

        inference_time = (
            time.perf_counter() - start_time
        ) * 1000

        vehicles = []

        for box in results.boxes:

            confidence = float(box.conf[0])

            # Use configurable confidence threshold
            if confidence < VEHICLE_CONFIDENCE:
                continue

            class_id = int(box.cls[0])

            label = self.model.names[class_id].lower()

            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            vehicles.append({

                "label": label,

                "confidence": confidence,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )

            })

        # Highest confidence detections first
        vehicles.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return (
            vehicles,
            len(vehicles),
            inference_time
        )

    def draw(self, image, vehicles):
        """
        Draw detected vehicles.
        """

        for i, vehicle in enumerate(vehicles, start=1):

            x1, y1, x2, y2 = vehicle["box"]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                VEHICLE_BOX_COLOR,
                2
            )

            # Show V1, V2, V3...
            cv2.putText(
                image,
                f"V{i}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                VEHICLE_BOX_COLOR,
                2
            )

        return image