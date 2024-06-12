import cv2


def start_cam():
    """
    Test the ports and returns a tuple with the available ports
    and the ones that are working.
    """
    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_EXPOSURE, 0.1)

    while True:
        cv2.imshow("Camera", camera.read()[1])
        cv2.waitKey(1)


if __name__ == "__main__":
    start_cam()
