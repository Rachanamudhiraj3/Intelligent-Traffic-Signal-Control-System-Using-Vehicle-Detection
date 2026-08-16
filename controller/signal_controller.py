
from config import (
    BASE_SIGNAL_TIME,
    TIME_PER_VEHICLE,
    MIN_SIGNAL_TIME,
    MAX_SIGNAL_TIME,
    EMERGENCY_SIGNAL_TIME
)


class SignalController:
    """
    Handles signal timing and green lane selection.
    """

    def __init__(self):
        pass

    def calculate_signal_time(self, vehicle_count, emergency=False):
        """
        Calculate signal time for a lane.
        """

        if emergency:
            return EMERGENCY_SIGNAL_TIME

        signal_time = BASE_SIGNAL_TIME + (vehicle_count * TIME_PER_VEHICLE)

        if signal_time < MIN_SIGNAL_TIME:
            signal_time = MIN_SIGNAL_TIME

        if signal_time > MAX_SIGNAL_TIME:
            signal_time = MAX_SIGNAL_TIME

        return signal_time

    def choose_green_lane(self, lane_data):
        """
        Select lane for GREEN signal.

        Priority:
        1. Emergency vehicle
        2. Maximum vehicle count
        """

        # Emergency gets highest priority
        for lane in lane_data:

            if lane["emergency"]:

                return lane["name"]

        # Otherwise choose highest traffic lane
        max_lane = max(
            lane_data,
            key=lambda x: x["vehicles"]
        )

        return max_lane["name"]

    def update_signal_status(self, lane_data):
        """
        Update RED/GREEN status for all lanes.
        """

        green_lane = self.choose_green_lane(lane_data)

        for lane in lane_data:

            lane["signal"] = "GREEN" if lane["name"] == green_lane else "RED"

        return lane_data, green_lane