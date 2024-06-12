import glob
import os

import cv2
import numpy as np

BOARD_DIMS = (7, 5)
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.03
DICT_SIZE = cv2.aruco.DICT_4X4_1000

FOLDER = os.path.dirname(__file__)


def detect_pose(image, camera_matrix, dist_coeffs):
    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_SIZE)
    board = cv2.aruco.CharucoBoard((BOARD_DIMS[1], BOARD_DIMS[0]), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(undistorted_image, dictionary, parameters=params)

    # If at least one marker is detected
    if len(marker_ids) > 0:
        # Interpolate CharUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, undistorted_image, board
        )

        # If enough corners are found, estimate the pose
        if charuco_retval:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
            )

            # If pose estimation is successful, draw the axis
            if retval:
                cv2.drawFrameAxes(undistorted_image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=5)
    return undistorted_image


# Load calibration data
camera_matrix = np.load(os.path.join(FOLDER, "calibration_data", "calibration_camera_matrix.npy"))
dist_coeffs = np.load(os.path.join(FOLDER, "calibration_data", "calibration_dist_coeffs.npy"))

# Iterate through PNG images in the folder
images = glob.glob(os.path.join(FOLDER, "calibration_data", "*.png"))
images.sort()  # Ensure files are in order

for image_file in images:
    # Load an image
    image = cv2.imread(image_file)

    # Detect pose and draw axis
    pose_image = detect_pose(image, camera_matrix, dist_coeffs)

    # Show the image
    cv2.imshow("Pose Image", pose_image)
    cv2.waitKey(0)
