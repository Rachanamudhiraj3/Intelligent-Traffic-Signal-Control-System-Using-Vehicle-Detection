import cv2
import numpy as np

from config import (
    WINDOW_NAME,
    FONT,
    TITLE_SCALE,
    TEXT_SCALE,
    TITLE_THICKNESS,
    TEXT_THICKNESS,
    WHITE,
    YELLOW,
    CYAN,
    GREEN,
    RED
)


class Display:

    def __init__(self):
        self.window_name = WINDOW_NAME

    def draw_lane_information(
        self,
        image,
        lane_name,
        vehicle_count,
        signal_time,
        signal_status,
        emergency=False,
        confidence=0.0,
        inference_time=0.0
    ):
        """
        Draw lane information.
        """

        cv2.putText(
            image,
            lane_name,
            (10, 25),
            FONT,
            TITLE_SCALE,
            WHITE,
            TITLE_THICKNESS
        )

        cv2.putText(
            image,
            f"Vehicles : {vehicle_count}",
            (10, 55),
            FONT,
            TEXT_SCALE,
            CYAN,
            TEXT_THICKNESS
        )

        cv2.putText(
            image,
            f"Signal Time : {signal_time}s",
            (10, 85),
            FONT,
            TEXT_SCALE,
            YELLOW,
            TEXT_THICKNESS
        )

        if emergency:

            cv2.putText(
                image,
                f"Emergency : YES ",
                (10, 115),
                FONT,
                TEXT_SCALE,
                RED,
                TEXT_THICKNESS
            )

        else:

            cv2.putText(
                image,
                "Emergency : NO",
                (10, 115),
                FONT,
                TEXT_SCALE,
                GREEN,
                TEXT_THICKNESS
            )

        if signal_status == "GREEN":

            color = GREEN

        else:

            color = RED

        cv2.putText(
            image,
            signal_status,
            (10, 180),
            FONT,
            TITLE_SCALE,
            color,
            TITLE_THICKNESS
        )

        return image

    def create_dashboard(
        self,
        lane1,
        lane2,
        lane3,
        lane4
    ):

        top = np.hstack((lane1, lane2))

        bottom = np.hstack((lane3, lane4))

        dashboard = np.vstack((top, bottom))

        dashboard = cv2.resize(
            dashboard,
            (1400, 800)
        )

        return dashboard

    def show(self, frame):
        """
        Display final output.
        Press any key to close.
        """

        cv2.imshow(
            self.window_name,
            frame
        )

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def close(self):

        cv2.destroyAllWindows()