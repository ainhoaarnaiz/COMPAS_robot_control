"""
This script uses machine vision to detect contours of objects (tiles) on a flat bed and obtain their world space coordinates.

Homography is used to convert pixel space coordinates to world space coordinates, meaning that the objects are
assumed to all be flat and laying on a flat surface.
The plane which constitutes said flat surface is defined by the corners of a QR code which is placed on it.

The corners of the QR code are pre-measured in world space using robot free motion.
The homographic matrix is calculated by opencv detecting the QR code automatically in the image.
"""

import time
import logging
from copy import deepcopy
from threading import Thread
from threading import Lock

from scipy.spatial import distance
import numpy as np
import cv2 as cv
from compas.geometry import Frame

from ui import UiManager
from ui import UserInput

# from compas_urt.perception import GenTlDevice
from calibration import OpenCVCalibrationData

# from compas_urt.design import RoundTile

LOG = logging.getLogger(__name__)

CALIBRATION_DATA = r"C:\Users\gcasas\eth\Workshops\workshop_iaac_2024\scripts\05_vision\calibration_data\calib_result_20240227-152441.dat"

# pre-measured corners of the QR code in world space.
# NOTE: order matters. first point is top left, then clockwise.
QR_CORNERS_PLANE = np.array(
    [
        [0.507, -0.285, 0.00],
        [0.436, -0.286, 0.00],
        [0.435, -0.215, 0.00],
        [0.507, -0.213, 0.00],
    ],
    dtype="float32",
).reshape((4, 1, 3))


def draw_text(image, text, pos, scale, color=(0, 240, 0), with_background=False):
    font = cv.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv.getTextSize(text, font, scale, 1)
    text_w, text_h = text_size
    cx, cy = pos
    if with_background:
        cv.rectangle(image, pos, (cx + text_w, cy + text_h), (0, 0, 0), -1)
    cv.putText(image, text, (cx, cy), font, scale, color, 1, cv.LINE_AA)


class ContourFinder(Thread):
    """An OpenCV based program which detects round tiles in the image and returns Tile objects."""

    def __init__(
        self,
        device_number: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ui_manager = UiManager("Display")
        self.device = cv.VideoCapture(device_number)
        # Azure Kinect settings at IAAC
        self.device.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.device.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.device.set(cv.CAP_PROP_FPS, 30)
        self.device.set(cv.CAP_PROP_EXPOSURE, 0.1)

        self.c_data = OpenCVCalibrationData.from_file(CALIBRATION_DATA)
        self.qr_detector = cv.QRCodeDetector()
        self.h_mat = None
        self.scaling_factor = None
        self.current_snapshot = None
        self.lock = Lock()
        self.is_running = False

    def _init_gui(self):
        # must be called AFTER UiManager.start()
        self.ui_manager.add_trackbar("min_x", 2500, 100)
        self.ui_manager.add_trackbar("max_x", 2500, 1000)
        self.ui_manager.add_trackbar("min_y", 2500, 100)
        self.ui_manager.add_trackbar("max_y", 2500, 1000)
        self.ui_manager.add_key_tracker(ord("f"))  # look for qr and draw corners
        self.ui_manager.add_key_tracker(ord("x"))  # store points and stop looking
        self.ui_manager.add_key_tracker(ord("s"))  # take snapshot
        self.ui_manager.WAIT_FOR_INPUT_MS = 1000

    def run(self) -> None:
        self.ui_manager.start()
        self._init_gui()
        self.is_running = True
        LOG.debug("starting perception service")
        search_qr = True

        try:
            while self.is_running:
                image = self.device.read()[1]
                input = self.ui_manager.get_user_input()
                # TODO: Calibrate cam to be able to fix distortion
                # image = self._undistortify(image)

                if search_qr and input.key_x and self.h_mat is not None:
                    search_qr = False
                elif not search_qr and input.key_f:
                    search_qr = True

                if search_qr:
                    self._detect_qr(image)  # do before drawing anything on the image
                else:
                    p_contours, p_centroids, p_circles = self._find_elements(image, input)
                    tiles = self._elements_to_tiles(p_centroids, p_circles)
                    self._draw_contours(image, p_contours, p_circles)
                    self._draw_tile_info(image, p_centroids, tiles)
                    if input.key_s:
                        with self.lock:
                            self.current_snapshot = deepcopy(tiles)
                image = self._downscale(image, 80)
                self.ui_manager.show_image(image)
        finally:
            self.stop()
            LOG.debug("closing device")
            self.device.stop()
            LOG.debug("perception service stopped!")

    def _elements_to_tiles(self, centroids, circles):
        """Uses homography to convert pixel space information about detected contours to irl space.

        For each matching pair of centroid and circle:
        1. convert the centroid to homogenous coordinates
        2. multiply the homogenous coordinates by the homographic matrix, to get the irl coordinates
        3. convert the irl coordinates back to cartesian coordinates (by scaling by z)
        4. create a tile frame assuming z=0
        5. scale diameter of the pixel space circle by the pre-calculated scale factor to get the irl diameter
        6. create a Tile object with the frame and diameter

        Parameters
        ----------
        centroids : list of tuples
            list of pixel space centroids of detected contours
        circles : list of tuples
            list of pixel space circles of detected contours (minimum enclosing circle)

        Returns
        -------
        list of Tile(center_frame, diameter)
            Tile objects containing center_frame and diameter in world coordinates.

        """
        # multiply each of the centroids by the homographic matrix and return the result
        if self.h_mat is None:
            return

        tiles = []
        for center, circle in zip(centroids, circles):
            p_points = np.array([[center[0], center[1], 1.0]], dtype="float32").reshape((3, 1))
            converted = np.matmul(self.h_mat, p_points)
            converted = converted / converted[-1]  # homogenous => cartesian
            irl_x, irl_y, _ = converted
            frame = Frame.worldXY()
            frame.point = [irl_x, irl_y, 0.0]
            _, radius = circle
            irl_diameter = radius * 2.0 / self.scaling_factor
            tiles.append({"base_frame": frame, "diameter": irl_diameter})
        return tiles

    def _undistortify(self, image: np.ndarray):
        height, width = image.shape[:2]
        camera = self.c_data.camera_matrix
        dist = self.c_data.dist_coefficients
        new_camera, roi = cv.getOptimalNewCameraMatrix(camera, dist, (width, height), 1, (width, height))
        undistorted = cv.undistort(image, camera, dist, None, new_camera)

        # undistorted image will likely be smaller
        x, y, w, h = roi
        return undistorted[y : y + h, x : x + w]

    def _detect_qr(self, image: np.ndarray):
        # https://temugeb.github.io/python/computer_vision/2021/06/15/QR-Code_Orientation.html
        ret_qr, points = self.qr_detector.detect(image)

        if not ret_qr:
            LOG.debug("no QR code found!")
            self.h_mat = None
            return None

        # just so we feel we've achieved something
        # also, the order of the points in pixel space has to match the order of irl coordinates defined below.
        for index, point in enumerate(points[0]):
            p_x, p_y = point.astype(int)
            draw_text(image, f"#{index}", (p_x, p_y), 0.5)
            cv.circle(image, point.astype(int), 3, (0, 0, 255), -1)

        # find the H matrix which converts from pixel space (plane defined by the 4 corners of the qr code in pixel space)
        # to the plane defined by the 4 corners of the qr code in its coordinate space.
        self.h_mat, _ = cv.findHomography(points, QR_CORNERS_PLANE)

        # irresponsibly estimate the pixel to meters scaling factor
        # https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
        pixel_dist = distance.euclidean(points[0][0], points[0][3])
        qr_width_m = 0.07343  # qr width 73mm
        self.scaling_factor = pixel_dist / qr_width_m
        print("found QR and h mat")

    def _downscale(self, image, scale_percent=100):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv.resize(image, dim, interpolation=cv.INTER_AREA)

    def _draw_tile_info(self, image, centroids, tiles):
        if centroids is None or tiles is None:
            LOG.debug("no centroids. exiting..")
            return

        for pixel_c, tile in zip(centroids, tiles):
            # visualize center point
            cv.circle(image, pixel_c, 3, (0, 0, 255), -1)
            cx, cy = pixel_c
            text = f"{tile['diameter']:.3f}"
            draw_text(image, text, (cx, cy + 5), scale=0.5, color=(255, 255, 0))

    def _draw_contours(self, image, contours, circles):
        """Visualizes pixel space information about found elements. This is useful for debugging and development."""
        cv.drawContours(image, contours, -1, (0, 255, 0), 3)
        for circle in circles:
            center, radius = circle
            cx, cy = center
            cv.circle(image, (int(cx), int(cy)), int(radius), (0, 255, 255))

    def _find_elements(self, image: np.ndarray, user_input: UserInput):
        """Finds elements (tiles) in image and returns pixel space information about them.

        1. roi is determined from user input and visualized
        2. min and max area in pixel value are (somewhat arbitrarily) determined to filter unwanted noise
        3. an appropriate thresholding value is determined using [the OTSU method](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
        4. contours are found in image and filtered using the above values, and from which these are calculated:
            - centroid in pixel space
            - minimum enclosing circle

        """
        start_row, end_row, start_col, end_col = user_input.min_x, user_input.max_x, user_input.min_y, user_input.max_y
        roi_area = (end_row - start_row) * (end_col - start_col)
        max_area = roi_area * 0.05  # rule of thumb: 5% of roi area
        min_area = roi_area * 0.001  # rule of thumb: 0.1% of roi area

        # start_row, end_row, start_col, end_col = 133, 1243, 78, 787

        cv.rectangle(image, (start_row, start_col), (end_row, end_col), (0, 255, 0))

        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cropped = gray_img[start_col:end_col, start_row:end_row]
        cropped = cv.GaussianBlur(cropped, (5, 5), 0)
        _, threshold = cv.threshold(cropped, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        offset = (start_row, start_col)  # offset from roi to original values
        contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE, offset=offset)
        centroids = []
        circles = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > max_area or area < min_area:
                continue
            circles.append(cv.minEnclosingCircle(cnt))
            m = cv.moments(cnt)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            centroids.append((cx, cy))

        return contours, centroids, circles

    def get_elements(self):
        """Get a the latest captured collection of element (tile) if availble, else None."""
        if self.current_snapshot:
            return self.current_snapshot

    def stop(self):
        LOG.debug("stopping perception service")
        self.is_running = False
        self.is_alive = False


def main():
    p = ContourFinder(1)
    p.start()

    try:
        while p.is_alive:
            elements = p.get_elements()
            if elements:
                print(f"got snapshot containing {len(elements)} elements")
            time.sleep(1)
    finally:
        p.stop()


if __name__ == "__main__":
    main()
