from dataclasses import dataclass
from copy import deepcopy
from typing import Tuple

import numpy
import cv2


@dataclass
class UserInput:
    """Stores user input state."""

    pass


class UiManager:
    """
    Wraps the windowing and user input functionality of openCV
    """

    WAIT_FOR_INPUT_MS = 10

    # key_unicode: user_input.attr_name
    KEY_FLAGS = {
        ord("q"): "should_exit",
        ord("s"): "should save",
    }

    def __init__(self, window_name="Window"):
        self.window_name = window_name
        self.user_input = UserInput()

    def get_user_input(self) -> UserInput:
        """
        Wait for user in put WAIT_FOR_USER
        :return:
        """
        self.reset_flags()
        k = cv2.waitKey(self.WAIT_FOR_INPUT_MS)
        if k != -1:
            attr_name = UiManager.KEY_FLAGS.get(k, None)
            if attr_name:
                setattr(self.user_input, attr_name, True)

        return self.user_input

    def reset_flags(self):
        for attr_name in self.KEY_FLAGS.values():
            setattr(self.user_input, attr_name, False)

    def start(self):
        """
        Start an OpenCV window and setup control callbacks
        :return:
        """
        cv2.namedWindow(self.window_name)

    def _is_window_closed(self):
        try:
            return cv2.getWindowProperty(self.window_name, 0) == -1
        except cv2.error:
            return True

    def add_trackbar(self, name, max_value, default_value):
        """Add a trackbar to the gui"""

        def on_change(value):
            setattr(self.user_input, name, value)

        on_change(default_value)
        cv2.createTrackbar(name, self.window_name, default_value, max_value, on_change)

    def add_key_tracker(self, key_code):
        """Track a key press for the given key code.

        A key press will be recorded as boolean in user_input.key_{chr(key_code)}.
        For example for the key 's' with code 115 the flag will be 'user_input.key_s'.

        Parameters
        ----------
        key_code : int
            The unicode value of the desired key, as returned by the `ord` builtin.

        Returns
        -------
        str
            The attribute name which will flag pressing of this key in user_input.

        """
        key_flag_attr = f"key_{chr(key_code)}"
        UiManager.KEY_FLAGS[key_code] = key_flag_attr
        return key_flag_attr

    def draw_selected_tiles(self, on_img: numpy.ndarray, tiles: Tuple[numpy.ndarray]) -> None:
        """
        Draw the found contours on top of the given image
        :param on_img: image to draw on
        :param tiles: the contours to draw
        """
        img_copy = deepcopy(on_img)
        cv2.drawContours(img_copy, tiles, -1, (0, 255, 0), 3)
        self.show_image(img_copy)

    def show_image(self, image):
        cv2.imshow(self.window_name, image)
        cv2.waitKey(20)  # without this, window just freezes when image is updated
