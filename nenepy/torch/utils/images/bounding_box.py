import numpy as np

from PIL import ImageFont, ImageDraw, Image

from nenepy.torch.utils.images import Color

DEFAULT_FONT_NAME = "UbuntuMono-R.ttf"
DEFAULT_FONT_SIZE = 12
DEFAULT_FONT = ImageFont.truetype(DEFAULT_FONT_NAME, DEFAULT_FONT_SIZE)


class BoundingBox:

    @staticmethod
    def draw_boxes(image, boxes, scores=None, indexes=None, label_names=None, threshold=0.0):
        """

        Args:
            image (np.ndarray):
            boxes (np.ndarray or list):
            scores (np.ndarray or list):
            indexes (np.ndarray or list):
            label_names (np.ndarray or list):
            threshold (float):

        Returns:
            np.ndarray:

        Shapes:
            -> [H, W, 3]
            -> [N, 4]
            -> [N]
            -> [N]
            -> [N]

        """
        image = Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)

        N = len(boxes)
        if scores is None:
            scores = [1] * N
        if indexes is None:
            indexes = [0] * N
        if label_names is None:
            label_names = ["?"] * N

        for i in range(N):
            box = boxes[i]
            score = scores[i]
            index = indexes[i]
            label_name = label_names[i]
            color = Color.index_to_rgb(index)

            if score < threshold:
                continue

            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=tuple(color), width=3)
            text_w, text_h = DEFAULT_FONT.getsize(label_name)
            draw.rectangle([box[0], box[1], box[0] + text_w, box[1] + text_h], fill=tuple(color))
            draw.text((box[0], box[1]), label_name, font=DEFAULT_FONT, fill='white')

        return np.array(image)
