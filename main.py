import cv2
from config import LANE_IMAGES
from utils.image_processing import ImageProcessor
from detectors.vechile_detector import VehicleDetector
from detectors.emergency_detector import EmergencyDetector
from controller.signal_controller import SignalController
from ui.display import Display
from utils.metrics import Metrics
from utils.graphs import Graphs


# INITIALIZE MODULES
processor = ImageProcessor()

vehicle_detector = VehicleDetector()

emergency_detector = EmergencyDetector()

signal_controller = SignalController()

display = Display()

metrics = Metrics()

graphs = Graphs()

# PROCESS ONE LANE
def process_lane(lane_name, image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load {image_path}"
        )

    image = processor.preprocess(image)

    vehicles, vehicle_count, vehicle_time = vehicle_detector.detect(
        image
    )

    image = vehicle_detector.draw(
        image,
        vehicles
    )

    (
        emergency_detected,
        emergency_box,
        emergency_confidence,
        emergency_time
    ) = emergency_detector.detect(image)

    image = emergency_detector.draw(
        image,
        emergency_detected,
        emergency_box
    )

    signal_time = signal_controller.calculate_signal_time(
        vehicle_count,
        emergency_detected
    )

    lane = {

        "name": lane_name,

        "image": image,

        "vehicles": vehicle_count,

        "signal_time": signal_time,

        "emergency": emergency_detected,

        "emergency_confidence": emergency_confidence,

        "vehicle_inference": vehicle_time,

        "emergency_inference": emergency_time

    }

    return lane

# PROCESS ALL LANES
lanes = []

for lane_name, image_path in LANE_IMAGES.items():

    lane = process_lane(
        lane_name,
        image_path
    )

    lanes.append(lane)

# DECIDE GREEN SIGNAL
lanes, green_lane = signal_controller.update_signal_status(
    lanes
)


print("\n========================================")
print(" GREEN SIGNAL :", green_lane)
print("========================================\n")

# Ground truth for the demo images
actual_emergency = {
    "Lane 1": False,
    "Lane 2": True,
    "Lane 3": False,
    "Lane 4": False
}

# DRAW INFORMATION
for lane in lanes:

    lane["image"] = display.draw_lane_information(

        lane["image"],

        lane["name"],

        lane["vehicles"],

        lane["signal_time"],

        lane["signal"],

        lane["emergency"],

        lane["emergency_confidence"],

        lane["vehicle_inference"] + lane["emergency_inference"]

    )


# CREATE FINAL DASHBOARD
final_frame = display.create_dashboard(

    lanes[0]["image"],

    lanes[1]["image"],

    lanes[2]["image"],

    lanes[3]["image"]

)

# PRINT RESULTS
print("============== LANE DETAILS ==============\n")

for lane in lanes:

    print(f"{lane['name']}")

    print(f"Vehicles           : {lane['vehicles']}")

    print(f"Emergency          : {lane['emergency']}")

    print(f"Signal Time        : {lane['signal_time']} sec")

    print(f"Signal             : {lane['signal']}")

    print(f"Vehicle Detection  : {lane['vehicle_inference']:.2f} ms")

    print(f"Emergency Detection: {lane['emergency_inference']:.2f} ms")

    print("-----------------------------------------")

# PERFORMANCE
total_vehicle_time = sum(
    lane["vehicle_inference"] for lane in lanes
)

total_emergency_time = sum(
    lane["emergency_inference"] for lane in lanes
)

total_time = total_vehicle_time + total_emergency_time

average_time = total_time / 4

fps = 1000 / average_time if average_time > 0 else 0

print("\n============== PERFORMANCE ==============")

print(f"Total Inference Time : {total_time:.2f} ms")

print(f"Average Per Lane     : {average_time:.2f} ms")

print(f"Estimated FPS        : {fps:.2f}")

print("=========================================\n")


print("=========================================\n")
metrics.print_model_metrics(
    "results.csv"
)

# DISPLAY OUTPUTs
final_frame = cv2.resize(
    final_frame,
    (1400,800)
)

display.show(final_frame)


graphs.plot_model_evaluation()

graphs.plot_comparison()