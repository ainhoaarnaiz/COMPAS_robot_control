import glob
import os

import cv2
import numpy as np

FOLDER = os.path.dirname(__file__)

BOARD_DIMS = (7, 5)
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.03
DICT_SIZE = cv2.aruco.DICT_4X4_1000

# Initialize arrays to store object points and image points from all images.
all_charuco_corners = []
all_charuco_ids = []
all_detected_corners = []
all_detected_ids = []

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

# Prepare object points based on the ChArUco board dimensions
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Load dictionary and parameters
dictionary = cv2.aruco.getPredefinedDictionary(DICT_SIZE)
board = cv2.aruco.CharucoBoard((BOARD_DIMS[1], BOARD_DIMS[0]), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
parameters = cv2.aruco.DetectorParameters()

imboard = board.generateImage((BOARD_DIMS[1] * 280, BOARD_DIMS[0] * 280))
cv2.imwrite(os.path.join(FOLDER, "charuco_board.png"), imboard)


# Load your images (update the path as needed)
images = glob.glob(os.path.join(FOLDER, "calibration_data", "*.png"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)

    board_corners, board_corner_ids, markers, marker_ids = detector.detectBoard(gray)
    marker_corners, marker_ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    # If at least one marker is detected...
    if len(marker_ids) > 0:
        # ...interpolate the ChArUco corners
        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

        # if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
        if charuco_retval:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

        # Refine the detected corners
        cv2.aruco.refineDetectedMarkers(gray, board, marker_corners, marker_ids, rejectedImgPoints)

        cv2.drawChessboardCorners(img, (BOARD_DIMS[1] - 1, BOARD_DIMS[0] - 1), charuco_corners, True)
        cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids, borderColor=(0, 255, 255))

    cv2.imshow("Calibration", img)
    key = cv2.waitKey(10000)
    if key == 27:
        cv2.destroyAllWindows()
        break

# Calibrate the camera using the detected corners
ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners, all_charuco_ids, board, gray.shape[::-1], None, None
)

print("RMS:\n", ret)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

np.save(os.path.join(FOLDER, "calibration_data", "calibration_camera_matrix.npy"), mtx)
np.save(os.path.join(FOLDER, "calibration_data", "calibration_dist_coeffs.npy"), dist)
