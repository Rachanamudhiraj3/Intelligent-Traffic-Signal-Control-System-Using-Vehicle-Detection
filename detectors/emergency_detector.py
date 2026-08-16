import time
import cv2
from ultralytics import YOLO

from config import (
    EMERGENCY_MODEL_PATH,
    EMERGENCY_CONFIDENCE,
    EMERGENCY_BOX_COLOR
)


class EmergencyDetector:
    """
    Detect emergency vehicles using the trained YOLO model.
    """

    def __init__(self):

        print("Loading Emergency Detection Model...")

        self.model = YOLO(str(EMERGENCY_MODEL_PATH))

        print("Emergency Model Loaded Successfully!\n")

    def detect(self, image):
        """
        Detect emergency vehicles.

        Returns:
            emergency_detected (bool)
            emergency_box (tuple or None)
            confidence (float)
            inference_time (float)
        """

        start_time = time.perf_counter()

        results = self.model(
            image,
            verbose=False
        )[0]

        inference_time = (
            time.perf_counter() - start_time
        ) * 1000

        emergency_detected = False
        emergency_box = None
        best_confidence = 0.0

        for box in results.boxes:

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            label = self.model.names[class_id].lower()


            # Ignore weak detections
            if confidence < EMERGENCY_CONFIDENCE:
                continue

            # Accept only emergency class
            if label != "emergency":
                continue

            if confidence > best_confidence:

                best_confidence = confidence

                emergency_detected = True

                emergency_box = tuple(
                    map(int, box.xyxy[0])
                )

        return (
            emergency_detected,
            emergency_box,
            best_confidence,
            inference_time
        )

    def draw(
        self,
        image,
        emergency_detected,
        emergency_box
    ):
        """
        Draw emergency bounding box.
        """

        if not emergency_detected or emergency_box is None:
            return image

        x1, y1, x2, y2 = emergency_box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            EMERGENCY_BOX_COLOR,
            3
        )

        cv2.putText(
            image,
            "EMERGENCY",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            EMERGENCY_BOX_COLOR,
            2
        )

        return image