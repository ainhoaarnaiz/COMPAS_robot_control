import glob
import os

import cv2
import numpy as np
import roslibpy
from scipy.spatial import distance

FOLDER = os.path.dirname(__file__)

# Define region of interest
MIN_X = 229
MAX_X = 1086
MIN_Y = 131
MAX_Y = 551
IMAGE_SOURCE = "camera"  # "camera" to get video real time, or "file" for testing with captured image frames
CAMERA_DEVICE_NUMBER = 1  # 0 for built-in camera, 1 for external camera
QR_DIAGONAL_M = 0.63  # QR Code width 63cm
QR_REAL_WORLD_POINTS = (
    np.array(  # Define corresponding points in real-world coordinates. These were measured with the robot.
        [
            [-0.3584, 0.2853],  # 1st point, QR Text="4", 4th QR,  index=0   # index 12 [12-15]
            [-0.3550, -0.2996],  # 2nd point, QR Text="1", 1st QR, index=1   # index 1 [0-3]
            [-0.5928, -0.3000],  # 3rd point, QR Text="3", 2nd QR, index=2   # index 6 [4-7]
            [-0.5965, 0.2847],  # 4th point, QR Text="2", 3rd QR,  index=3   # index 11 [8-11]
        ],
        dtype="float32",
    )
)

camera_matrix = np.load(os.path.join(FOLDER, "calibration_data", "calibration_camera_matrix.npy"))
dist_coeffs = np.load(os.path.join(FOLDER, "calibration_data", "calibration_dist_coeffs.npy"))

if IMAGE_SOURCE == "camera":
    # Azure Kinect settings at IAAC
    device = cv2.VideoCapture(CAMERA_DEVICE_NUMBER)
    device.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    device.set(cv2.CAP_PROP_FPS, 30)
    device.set(cv2.CAP_PROP_EXPOSURE, 0.1)

    def get_image_frames():
        while True:
            yield device.read()[1]

else:
    images = glob.glob(os.path.join(FOLDER, "qr", "*.png"))

    def get_image_frames():
        for fname in images:
            yield cv2.imread(fname)


# Connect to ROS to publish detected tile positions
ros = roslibpy.Ros(host="localhost", port=9090)
ros.run()

topic = roslibpy.Topic(ros, "/detected_tiles", "geometry_msgs/Point")
topic.advertise()

for img in get_image_frames():

    # # Step 0: Undistort the image
    # height, width = img.shape[:2]
    # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    #     camera_matrix, dist_coeffs, (width, height), 0.5, (width, height)
    # )
    # undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # # undistorted image will likely be smaller
    # x, y, w, h = roi
    # img = undistorted[y : y + h, x : x + w]

    # Step 1: Detect QR Code corners in the image
    qr_detector = cv2.QRCodeDetector()
    ret, decodedTexts, image_points, _ = qr_detector.detectAndDecodeMulti(img)
    image_points = image_points.reshape(-1, 2)
    qr_data = {}
    for qr_index, text in enumerate(decodedTexts):
        qr_data[text] = image_points[qr_index * 4 : (qr_index + 1) * 4, :]
    if sorted(decodedTexts) != ["1", "2", "3", "4"]:
        # Some QR codes failed to detect, skip frame
        print("Skipping frame, QR codes not detected")
        continue

    pixel_dist = distance.euclidean(qr_data["3"][2], qr_data["4"][0])
    scaling_factor = pixel_dist / QR_DIAGONAL_M
    new_image_points = np.array([qr_data["4"][0], qr_data["1"][1], qr_data["3"][2], qr_data["2"][3]])

    # Step 2: Compute the Homography Matrix
    H, _ = cv2.findHomography(new_image_points, QR_REAL_WORLD_POINTS)

    # Step 3: Find circles in the image (within region of interest defined by MIN_X, MAX_X, MIN_Y, MAX_Y)
    start_row, end_row, start_col, end_col = MIN_X, MAX_X, MIN_Y, MAX_Y
    roi_area = (end_row - start_row) * (end_col - start_col)
    max_area = roi_area * 0.2  # rule of thumb: 20% of roi area
    min_area = roi_area * 0.001  # rule of thumb: 0.1% of roi area

    cv2.rectangle(img, (start_row, start_col), (end_row, end_col), (0, 255, 0))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped = gray[start_col:end_col, start_row:end_row]
    cropped = cv2.GaussianBlur(cropped, (5, 5), 0)
    _, threshold = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    offset = (start_row, start_col)  # offset from roi to original values

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE, offset=offset)
    centroids = []
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area or area < min_area:
            continue
        try:
            m = cv2.moments(cnt)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            circles.append(cv2.minEnclosingCircle(cnt))
            centroids.append((cx, cy))
        except ZeroDivisionError:
            # Too small
            pass

    # Draw detected circles on the image
    for circle in circles:
        center, radius = circle
        cx, cy = center
        cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 255, 255))

    # Step 4: Transform centrois from image space to real-world space
    for centroid, circle in zip(centroids, circles):
        point = np.array([centroid], dtype="float32")
        point = np.array([point])
        real_world_point = cv2.perspectiveTransform(point, H).reshape(-1, 2)

        _circle_center, circle_radius = circle
        diameter_m = circle_radius * 2.0 / scaling_factor

        text = f"({real_world_point[0][0]*1000:.3f}, {real_world_point[0][1]*1000:.3f})"
        cv2.putText(img, text, (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("Detected tile. Real World coords:", real_world_point)

        # Publish detected tile centroid to ROS topic
        message = roslibpy.Message({"x": float(real_world_point[0][0]), "y": float(real_world_point[0][1]), "z": 0.0})
        topic.publish(message)

    cv2.imshow("Tile detection", img)
    key = cv2.waitKey(3000000)
    if key == 27:
        cv2.destroyAllWindows()
        break
