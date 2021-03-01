import numpy as np
import cv2


class Image:

    def __init__(self, image):
        """

        Args:
            image (ImageEvent):

        """
        self.image = self.encoded_image_string_to_numpy(image.encoded_image_string)
        self.step = image.step
        self.wall_time = image.wall_time
        self.width, self.height = self.image.shape[:2]

    @staticmethod
    def encoded_image_string_to_numpy(encoded_image_string):
        """

        Args:
            encoded_image_string (bytes):

        Returns:
            np.ndarray:

        """
        image = np.frombuffer(encoded_image_string, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
