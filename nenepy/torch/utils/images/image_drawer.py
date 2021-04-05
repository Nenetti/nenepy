from collections import OrderedDict

import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nenepy.torch.utils.images.color import Color

DEFAULT_FONT_NAME = "UbuntuMono-R.ttf"
DEFAULT_FONT_SIZE = 25
DEFAULT_FONT = ImageFont.truetype(DEFAULT_FONT_NAME, 22)


class ImageDrawer:

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def to_font(cls, font_size):
        """
        Generate Font

        Args:
            font_size (int):

        Returns:
            FreeTypeFont:

        """
        return ImageFont.truetype(DEFAULT_FONT_NAME, font_size)

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================

    @classmethod
    def _make_information_texts(cls, labels, score, class_names, threshold):
        """

        Args:
            labels (np.ndarray):
            score (np.ndarray):
            class_names (np.ndarray):

        Returns:
            (str, str):

        """

        # ----- Display Text ----- #
        gt_labels = np.array(class_names)[labels == 1]
        gt_indexes = np.where(labels == 1)[0]
        gt_predict_labels = [f"{class_names[k]}: {score[k]:.2f}" for k in gt_indexes]
        gt_predict_label_text = f"Positive Classification          : {', '.join(gt_predict_labels)}"
        error_indexes = np.where((score > threshold).astype(np.int32) - (labels == 1).astype(np.int32) == 1)[0]
        error_labels = [f"{class_names[k]}: {score[k]:.2f}" for k in error_indexes]
        error_label_text = f"Negative Classification (p > 0.1): {', '.join(error_labels)}"

        gt_label_text = f"GT  : {', '.join(gt_labels)}"

        return gt_label_text, gt_predict_label_text, error_label_text

    @classmethod
    def _make_information_iou_texts(cls, labels, iou, class_names):
        """

        Args:
            labels (np.ndarray):
            score (np.ndarray):
            class_names (np.ndarray):

        Returns:
            (str, str):

        """

        # ----- Sort Score ----- #

        # ----- Display Text ----- #
        indexes = np.where(labels == 1)[0]
        label_names = [f"{class_names[k]}: {iou[k]:.2f}" for k in indexes]
        predict_label_text = f"IOU : {', '.join(label_names)}"

        return predict_label_text

    @classmethod
    def _make_information_frame(cls, texts, font=None, font_size=-1, font_color=(255, 255, 255), indent_space=10):
        """

        Args:
            texts (list[str]):
            font (PIL.ImageFont.FreeTypeFont):
            font_size (int):
            font_color ((int, int, int)):
            indent_space (int):

        Returns:
            np.ndarray:

        Shapes:
            Args -> [3, H, W]

        """
        # ----- Font ----- #
        if font is None:
            font = DEFAULT_FONT
        if font_size == -1:
            font_size = DEFAULT_FONT_SIZE

        # ----- Text Size ----- #
        text_width = max([cls._calc_text_size(text, font)[0] for text in texts])
        # ----- Create and Draw Information ----- #
        label_text_area = Image.fromarray(np.zeros(shape=(len(texts) * font_size, text_width, 3), dtype=np.uint8))
        label_text_draw = ImageDraw.Draw(label_text_area)

        for i, text in enumerate(texts):
            x = indent_space
            y = i * font_size
            label_text_draw.text(xy=(x, y), text=text, fill=font_color, font=font)

        label_text_area = np.array(label_text_area, dtype=np.float32) / 255.0
        return label_text_area.transpose((2, 0, 1))

    @classmethod
    def _make_grid_image(cls, image_dict, colors=None, max_n_column=-1, default_font=None, font_size=-1):
        """

        Args:
            image_dict (OrderedDict[str, np.ndarray]):
            colors (tuple or list or np.ndarray)
            max_n_column (int):
            default_font (PIL.ImageFont.FreeTypeFont):
            font_size (int):

        Returns:
            np.ndarray:

        Shapes:
            N * [3, H, W] -> [3, H, W]

        """
        if max_n_column == -1:
            max_n_column = len(image_dict)

        # ----- Color ----- #
        if colors is None:
            colors = [None] * len(image_dict)

        elif not isinstance(colors, np.ndarray):
            colors = np.array(colors)

        # ----- Font ----- #
        if default_font is None:
            default_font = default_font
        if font_size == -1:
            font_size = DEFAULT_FONT_SIZE

        # ----- Calculate grid row and column ----- #
        n_images = len(image_dict)
        n_column, n_rows = cls._calc_optimal_grid_size(n_images, max_n_column)

        # ----- etc ----- #
        images = list(image_dict.values())
        keys = list(image_dict.keys())
        _, _, width = images[0].shape

        # ------------------------------
        #
        # Make Grid
        #
        #
        out_image = [None] * n_rows

        for i in range(n_rows):

            # ----- Extract Row Image ----- #
            si = i * n_column
            ei = min((i + 1) * n_column, n_images)

            row_images = images[si:ei]
            row_keys = keys[si:ei]
            row_colors = colors[si:ei]

            raw_image_area = np.concatenate(row_images, axis=2)

            # ----- Create and Draw Information ----- #
            _, _, total_width = raw_image_area.shape
            image_name_area = np.zeros((font_size, total_width, 3), dtype=np.uint8)
            image_name_area = Image.fromarray(image_name_area)
            image_name_draw = ImageDraw.Draw(image_name_area)

            for k, text in enumerate(row_keys):
                # text_width = cls._calc_text_size(str(text), default_font)[0]
                # if text_width > width:
                #     max_text_length = max([len(str(key)) for key in keys])
                #     size = ((width * 2) // max_text_length) - 1
                #     font = cls.resize_font(size)
                # else:
                #     font = default_font
                font = default_font

                cls._draw_frame(
                    image_name_draw, text,
                    sxy=(width * k, 0), exy=(width * (k + 1) - 1, font_size - 1),
                    rect_color=row_colors[k], font=font,
                )

            # ----- Summarized Image and Information ----- #
            image_name_area = np.array(image_name_area, dtype=np.float32) / 255.0
            image_name_area = image_name_area.transpose((2, 0, 1))
            out_image[i] = np.concatenate([image_name_area, raw_image_area], axis=1)

        return cls._auto_fitting_concat_images(out_image, dim=1)

    @classmethod
    def _draw_frame(cls, pil_draw, text, sxy, exy,
                    font=None, font_color=(255, 255, 255),
                    rect_color=None, indent_space=10, color_margin=5,
                    background_color=(0, 0, 0), frame_color=(255, 255, 255)):
        """

        Args:
            pil_draw (PIL.ImageDraw.ImageDraw):
            text (str):
            sxy ((int, int)):
            exy ((int, int)):
            font (PIL.ImageFont.FreeTypeFont):
            font_color ((int, int, int)):
            rect_color ((int, int, int)):
            indent_space (int):
            color_margin (int):
            background_color ((int, int, int)):
            frame_color ((int, int, int)):

        """
        # ----- Font ----- #
        if font is None:
            font = DEFAULT_FONT

        sx, sy = sxy
        ex, ey = exy
        # ----- Draw Information Frame ----- #
        pil_draw.rectangle(
            xy=((sx, sy), (ex, ey)),
            fill=background_color,
            outline=frame_color,
            width=1
        )

        if rect_color is None:
            # ----- Draw Information Text ----- #
            pil_draw.text(
                xy=(sx + indent_space, sy),
                text=text,
                fill=font_color,
                font=font
            )

        else:
            color_size = (ey - sy) - (color_margin * 2)
            # ----- Draw Information Text ----- #
            pil_draw.text(
                xy=(sx + indent_space + color_margin + color_size + color_margin + 5, sy),
                text=text,
                fill=font_color,
                font=font
            )

            # ----- Draw Information Color ----- #
            pil_draw.rectangle(
                xy=(
                    (sx + indent_space + color_margin, sy + color_margin),
                    (sx + indent_space + color_margin + color_size, ey - color_margin)
                ),
                fill=tuple(rect_color),
            )

    @classmethod
    def _auto_fitting_concat_images(cls, images, dim):
        """

        Args:
            images (list[np.ndarray]):
            dim (int):

        Returns:
            np.ndarray:

        Shapes:
            example: ([N, C, H, W], dim=1) -> [C, H * N, W]

        """
        if len(images) == 1:
            return images[0]
        else:
            shape = images[0].shape
            is_same_shapes = [image.shape == shape for image in images[1:]]
            if False not in is_same_shapes:
                return np.concatenate(images, axis=1)

        # ----- Calc Image Shape ----- #
        target_shape = cls._calc_concat_shape(images, dim)
        target_shape = np.array(target_shape)

        # ----- Concat Image (Overwrite) ----- #
        out_image = np.zeros(shape=target_shape.tolist())
        target_dim_start_pixel = 0
        for image in images:

            # ----- Get Different Shape Dimensions ----- #
            image_shape = np.array(image.shape)
            different_size_dim = np.where(target_shape != image_shape)[0]

            # ----- Get Overwrite (Copy) Area ----- #
            image_area = out_image
            for d in different_size_dim:
                start = 0
                length = int(image_shape[d])
                if d == dim:
                    start = target_dim_start_pixel

                if d == 1:
                    image_area = image_area[:, start:start + length]
                elif d == 2:
                    image_area = image_area[:, :, start:start + length]
                else:
                    raise ValueError()

            # ----- Overwrite Tensor (Copy) ----- #
            image_area[...] = image
            target_dim_start_pixel += image_shape[dim]
        return out_image

    @staticmethod
    def _calc_optimal_grid_size(n_images, max_n_column):
        """
        Calculate optimal grid size.

        Args:
            n_images (int):
            max_n_column (int):

        Returns:
            (int, int):

        """
        n_column = min(n_images, max_n_column)
        if n_column < n_images:
            n_rows = np.ceil(n_images / n_column)
        else:
            n_rows = 1
        return int(n_column), int(n_rows)

    @staticmethod
    def _calc_concat_shape(tensors, dim):
        """

        Args:
            tensors (list[np.ndarray]):
            dim (int):

        Returns:
            list[int]:

        """
        max_shape = np.array(tensors[0].shape)

        cat_dim_size = 0
        for tensor in tensors:
            max_shape = np.maximum(max_shape, np.array(tensor.shape))
            cat_dim_size += tensor.shape[dim]

        max_shape[dim] = cat_dim_size

        return max_shape.tolist()

    @staticmethod
    def _calc_text_size(text, font):
        """

        Args:
            text (str):
            font (PIL.ImageFont.FreeTypeFont):

        Returns:
            (int, int):

        """
        return ImageDraw.Draw(Image.fromarray(np.zeros(shape=(1, 1)))).textsize(text, font)
