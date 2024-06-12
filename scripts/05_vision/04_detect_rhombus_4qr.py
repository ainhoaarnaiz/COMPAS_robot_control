import glob
import os

import cv2
import numpy as np
from scipy.spatial import distance
import roslibpy

FOLDER = os.path.dirname(__file__)

# Define region of interest
MIN_X = 265
MAX_X = 976
MIN_Y = 165
MAX_Y = 543
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

# Define color ranges in HSV
color_ranges = {
    "red": ((150, 100, 100), (200, 255, 255)),
    "orange": ((11, 100, 100), (20, 255, 255)),
    "yellow": ((21, 100, 100), (30, 255, 255)),
    "green": ((31, 100, 100), (70, 255, 255)),
    "blue": ((101, 100, 100), (130, 255, 255)),
    "purple": ((131, 100, 100), (160, 255, 255)),
    "pink": ((161, 100, 100), (170, 255, 255)),
    "white": ((0, 0, 200), (180, 50, 255)),
    "grey": ((0, 0, 51), (180, 50, 200)),
    "black": ((0, 0, 0), (180, 50, 50)),
}

def get_color_name(hsv_color):
    for color, (lower, upper) in color_ranges.items():
        if lower[0] <= hsv_color[0] <= upper[0] and lower[1] <= hsv_color[1] <= upper[1] and lower[2] <= hsv_color[2] <= upper[2]:
            return color
    return "unknown"

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
    print(FOLDER)
    images = glob.glob(os.path.join(FOLDER, "frames_4qr", "*.png"))

    def get_image_frames():
        for fname in images:
            print(fname)
            yield cv2.imread(fname)

# Connect to ROS to publish detected tile positions
ros = roslibpy.Ros(host="localhost", port=9090)
ros.run()

topic = roslibpy.Topic(ros, "/detected_tile", "geometry_msgs/Point")
topic.advertise()

topic2 = roslibpy.Topic(ros, "/tile_color", "std_msgs/String")
topic2.advertise()

topic3 = roslibpy.Topic(ros, "/tile_width", "std_msgs/String")
topic3.advertise()

topic4 = roslibpy.Topic(ros, "/tile_rotation", "std_msgs/String")
topic3.advertise()


for img in get_image_frames():

    widths = []
    centroids = []
    z_angles = []

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

    H, _ = cv2.findHomography(new_image_points, QR_REAL_WORLD_POINTS)

    start_row, end_row, start_col, end_col = MIN_X, MAX_X, MIN_Y, MAX_Y
    roi_area = (end_row - start_row) * (end_col - start_col)
    max_area = roi_area * 0.2  # rule of thumb: 20% of roi area
    min_area = roi_area * 0.001  # rule of thumb: 0.1% of roi area

    cv2.rectangle(img, (start_row, start_col), (end_row, end_col), (0, 255, 0))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    offset = (start_row, start_col)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped = gray[MIN_Y:MAX_Y, MIN_X:MAX_X]
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=offset)

    prev_centroid = [0,0]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area or area < min_area:
            continue
        try:
            # Find the minimum enclosing diamond
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Calculate centroid
            m = cv2.moments(approx)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            centroid = (cx, cy)

            if abs(centroid[0] - prev_centroid[0]) > 5 or abs(centroid[1] - prev_centroid[1]) > 5:
                prev_centroid = centroid
                continue
            else:
                # Calculate width and height of the diamond
                rect = cv2.boundingRect(approx)
                width = rect[2]
                height = rect[3]

                # Compute principal axis
                covariance_matrix = np.cov(approx.reshape(-1, 2).T)
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                # Calculate angle of the principal axis
                angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
                angle_deg = np.degrees(angle_rad)
                # Assuming height > width
                if angle_deg < 0:
                    angle_deg += 180
                # Convert angle to radians
                angle_rad = np.radians(angle_deg)

                # Draw contour of the diamond
                cv2.drawContours(img, [approx], 0, (255, 0, 0), 2)

                # # Draw width and height on image
                # cv2.putText(img, f"Width: {width:.2f}", (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             (255, 255, 255), 2)
                # cv2.putText(img, f"Height: {height:.2f}", (centroid[0] + 10, centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (255, 255, 255), 2)

                # Get color of the tile in BGR
                bgr_color = img[cy, cx]
                # Convert BGR to HSV
                hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
                # Get color name
                color_name = get_color_name(hsv_color)
                # Convert color to string
                color_text = f"Color: {color_name}"

                # Draw centroid
                cv2.circle(img, centroid, 5, (0, 0, 255), -1)

                # Calculate endpoint of the arrow line
                arrow_length = 50  # Length of the arrow line
                end_x = int(cx + arrow_length * np.cos(angle_rad))
                end_y = int(cy + arrow_length * np.sin(angle_rad))
                # Draw arrow line on the image
                arrow_color = (0, 255, 0)  # Green color
                arrow_thickness = 2
                cv2.arrowedLine(img, centroid, (end_x, end_y), arrow_color, arrow_thickness)

                # Draw color text on image
                cv2.putText(img, color_text, (centroid[0] + 10, centroid[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

                # Print width, height, centroid, and color
                print(f"Width: {width:.2f}, Height: {height:.2f}, Centroid: {centroid}, Color: {color_name}")

                prev_centroid = centroid

                widths.append(width)
                centroids.append((cx, cy))
                z_angles.append(angle_rad)


        except ZeroDivisionError:
            pass

         # Step 4: Transform centrois from image space to real-world space
        for centroid, width, z_angle in zip(centroids, widths, z_angles):
            point = np.array([centroid], dtype="float32")
            point = np.array([point])
            real_world_point = cv2.perspectiveTransform(point, H).reshape(-1, 2)

            # Transform rotation angle to real-world coordinates
            rotation_angle_real = np.arctan2(np.sin(z_angle) * H[0, 1] + np.cos(z_angle) * H[0, 0], 
                                            np.sin(z_angle) * H[1, 1] + np.cos(z_angle) * H[1, 0])
            # Convert angle to degrees
            rotation_angle_deg = np.degrees(rotation_angle_real)

            width_m = width / scaling_factor
            cv2.putText(img, f"Width: {width_m:.4f}", (centroid[0] + 10, centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        print(rotation_angle_deg)

        # Publish one tile to ROS topic
        message = roslibpy.Message({"x": float(real_world_point[0][0]), "y": float(real_world_point[0][1]), "z": 0.0})
        topic.publish(message)
        message2 = roslibpy.Message({"data": color_name})
        topic2.publish(message2)
        message3 = roslibpy.Message({"data": str(width_m)})
        topic3.publish(message3)
        message4 = roslibpy.Message({"data": str(rotation_angle_real)})
        topic4.publish(message4)


    cv2.imshow("Tile detection", img)
    key = cv2.waitKey(3000)
    if key == 27:
        cv2.destroyAllWindows()
        break
