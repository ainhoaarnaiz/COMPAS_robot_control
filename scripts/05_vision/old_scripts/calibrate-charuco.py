import time
import os
from copy import deepcopy
from argparse import ArgumentParser

import cv2 as cv
import numpy as np


BOARD_DIMS = (5, 7)
SQUARE_SIZE_M = 0.04
MARKER_SIZE_M = 0.03
DICT_SIZE = cv.aruco.DICT_4X4_1000
WINDOW = "Charuco Calibration"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "calibration_data")
DEVICE_ID = 1


def calibrate():
    device = cv.VideoCapture(DEVICE_ID)
    board, dictionary = create_board()
    # Generate a PNG of the board
    # imboard = board.generateImage((600, 500))
    # cv.imwrite(os.path.join(os.path.dirname(__file__), "charuco.png"), imboard)
    detector = cv.aruco.CharucoDetector(board)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    corners_list = []
    ids_list = []
    object_points = []
    image_points = []
    frame_list = []

    while True:
        image = device.read()[1]
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_size = grayscale.shape
        corners, corner_ids, markers, marker_ids = detector.detectBoard(grayscale)

        image_copy = deepcopy(image)
        ret = corner_ids is not None

        if ret:
            objp, img_points = board.matchImagePoints(corners, corner_ids)

            cv.aruco.drawDetectedMarkers(image_copy, markers, marker_ids, borderColor=(0, 255, 255))
            cv.aruco.drawDetectedCornersCharuco(image_copy, corners, corner_ids, cornerColor=(255, 255, 0))

        cv.imshow(WINDOW, image_copy)

        key = cv.waitKey(20)
        if key == 27:
            break
        elif key == ord("c"):
            # record frame and its detected markers
            if corner_ids is None or len(corner_ids) == 0:
                print("No ids detected!")
                continue
            cv.imwrite(os.path.join(RESULT_DIR, f"calibration_frame_{time.time()}.png"), image)

            corners_list.append(corners)
            ids_list.append(corner_ids)
            object_points.append(objp)
            image_points.append(img_points)
            frame_list.append(grayscale)

    # start calibration using collected data
    if not (corners_list or ids_list or frame_list):
        print("no calibration info captured")

    corners_list = [x for x in corners_list if len(x) >= 4]
    ids_list = [x for x in ids_list if len(x) >= 4]
    # Method 1
    ret, camera_matrix, dist_coeff, rvec, tvec = cv.aruco.calibrateCameraCharuco(
        corners_list, ids_list, board, image_size, None, None
    )

    print(f"ret:{ret}, camera_matrix:{camera_matrix}, dist_coeff:{dist_coeff}, rvec:{rvec}, tvec:{tvec}")

    save_coefficients(camera_matrix, dist_coeff, rvec, tvec, RESULT_DIR)
    # # Method 2
    # ret, camera_matrix, dist_coeff, rvec, tvec = cv.aruco.calibrateCameraCharuco(
    #     object_points, image_points, board, image_size, None, None
    # )

    # print(f"ret:{ret}, camera_matrix:{camera_matrix}, dist_coeff:{dist_coeff}, rvec:{rvec}, tvec:{tvec}")

    # save_coefficients(camera_matrix, dist_coeff, rvec, tvec, RESULT_DIR)


def create_board():
    d = cv.aruco.getPredefinedDictionary(DICT_SIZE)
    return cv.aruco.CharucoBoard(BOARD_DIMS, SQUARE_SIZE_M, MARKER_SIZE_M, d), d


def save_coefficients(mtx, dist, r_vecs, t_vecs, dir_path):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(dir_path, f"calib_result_{timestr}.dat")
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.write("R", r_vecs[0])
    cv_file.write("T", t_vecs[0])
    cv_file.release()


if __name__ == "__main__":
    calibrate()
