from collections import OrderedDict

import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nenepy.torch.utils.images import Mask
from nenepy.torch.utils.images.color import Color


class AbstractImageVisualizer:
    default_font_name = "UbuntuMono-R.ttf"
    default_font = ImageFont.truetype(default_font_name, 22)
    default_font_size = 25

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    @classmethod
    def resize_font(cls, font_size):
        return ImageFont.truetype(cls.default_font_name, font_size)

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
        gt_predict_label_text = f"Positive Classification: {', '.join(gt_predict_labels)}"
        error_indexes = np.where((score > threshold).astype(np.int32) - (labels == 1).astype(np.int32) == 1)[0]
        error_labels = [f"{class_names[k]}: {score[k]:.2f}" for k in error_indexes]
        error_label_text = f"Negative Classification: {', '.join(error_labels)}"

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
            font = cls.default_font
        if font_size == -1:
            font_size = cls.default_font_size

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
            default_font = cls.default_font
        if font_size == -1:
            font_size = cls.default_font_size

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

        class_color_space = font_size

        for i in range(n_rows):

            # ----- Extract Row Image ----- #
            si = i * n_column
            ei = min((i + 1) * n_column, n_images)

            row_images = images[si:ei]
            row_keys = keys[si:ei]
            row_colors = colors[si:ei]

            raw_image_area = np.concatenate(row_images, axis=2)

            # ----- Create and Draw Information ----- #
            n_rows = 1
            for text in row_keys:
                if "\n" in text:
                    splits = text.split("\n")
                    n_rows = max([n_rows, len(splits)])

            _, _, total_width = raw_image_area.shape
            # image_name_area = np.zeros((font_size * n_rows, total_width, 3), dtype=np.uint8)
            # image_name_area = Image.fromarray(image_name_area)
            # image_name_draw = ImageDraw.Draw(image_name_area)
            #
            row_frames = [None] * len(row_keys)
            height = (font_size * n_rows)
            for k, text in enumerate(row_keys):
                if row_colors[k] is not None:
                    row_frames[k] = cls._generate_cam_text_frame(text, width, height, tuple(row_colors[k]), frame_color=(255, 255, 255))
                else:
                    row_frames[k] = cls._generate_text_frame(text, width, height, frame_color=(255, 255, 255))
            image_name_area = np.concatenate(row_frames, axis=2)

            #     split_texts = text.split("\n")
            #     cls._draw_frame(
            #         image_name_draw,
            #         sxy=(width * k, 0), exy=(width * (k + 1) - 1, (font_size * len(split_texts)) - 2),
            #         frame_color=(255, 255, 255)
            #     )
            #     if row_colors[k] is not None:
            #         cls._draw_frame(
            #             image_name_draw,
            #             sxy=(1, 1), exy=(class_color_space - 1, (font_size * len(split_texts)) - 2),
            #             background_color=tuple(row_colors[k])
            #         )
            #     for m, split_text in enumerate(split_texts):
            #         cls._draw_text(
            #             image_name_draw,
            #             split_text,
            #             sxy=(width * k + class_color_space, font_size * m),
            #         )
            #
            # # ----- Summarized Image and Information ----- #
            # image_name_area = np.array(image_name_area, dtype=np.float32) / 255.0
            # image_name_area = image_name_area.transpose((2, 0, 1))
            out_image[i] = np.concatenate([image_name_area, raw_image_area], axis=1)

        return cls._auto_fitting_concat_images(out_image, dim=1)

    @classmethod
    def _generate_cam_text_frame(cls, text, width, height, class_color, background_color=None, frame_color=None):
        """

        Args:
            text (str):
            sxy ((int, int)):
            exy ((int, int)):
            frame_color (None or (int, int, int)):
            font_color ((int, int, int)):
            class_color ((int, int, int)):

        """
        font_size = cls.default_font_size
        pil_image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        image_draw = ImageDraw.Draw(pil_image)

        class_color_space = cls.default_font_size
        cls._draw_frame(
            image_draw,
            sxy=(0, 0), exy=(width - 1, height - 1),
            outline_color=frame_color
        )

        cls._draw_frame(
            image_draw,
            sxy=(0, 0), exy=(class_color_space - 1, height - 1),
            background_color=class_color, outline_color=frame_color,
        )

        for m, split_text in enumerate(text.split("\n")):
            cls._draw_text(
                image_draw,
                split_text,
                sxy=(class_color_space, font_size * m),
            )

        image = np.array(pil_image, dtype=np.float32) / 255.0
        return image.transpose((2, 0, 1))

    @classmethod
    def _generate_text_frame(cls, text, width, height, background_color=None, frame_color=None):
        """

        Args:
            text (str):
            width (int):
            height (int):
            background_color (None or (int, int, int)):
            frame_color (None or (int, int, int)):

        """
        font_size = cls.default_font_size
        pil_image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        image_draw = ImageDraw.Draw(pil_image)

        cls._draw_frame(
            image_draw,
            sxy=(0, 0), exy=(width - 1, height - 1),
            background_color=background_color,
            outline_color=frame_color,
        )

        for m, split_text in enumerate(text.split("\n")):
            cls._draw_text(
                image_draw,
                split_text,
                sxy=(0, font_size * m),
            )

        image = np.array(pil_image, dtype=np.float32) / 255.0
        return image.transpose((2, 0, 1))

    @classmethod
    def _draw_frame(cls, image_draw, sxy, exy, outline_width=1, background_color=None, outline_color=None):
        """

        Args:
            image_draw (PIL.ImageDraw.ImageDraw):
            sxy ((int, int)):
            exy ((int, int)):
            outline_color (None or (int, int, int)):

        """
        image_draw.rectangle(
            xy=(sxy, exy),
            fill=background_color,
            outline=outline_color,
            width=outline_width
        )

    @classmethod
    def _draw_text(cls, image_draw, text, sxy,
                   font=None, font_color=(255, 255, 255),
                   rect_color=None, indent_space=10, color_margin=5, ):
        """

        Args:
            image_draw (PIL.ImageDraw.ImageDraw):
            text (str):
            sxy ((int, int)):
            font (PIL.ImageFont.FreeTypeFont):
            font_color ((int, int, int)):
            rect_color ((int, int, int)):
            indent_space (int):
            color_margin (int):

        """
        # ----- Font ----- #
        if font is None:
            font = cls.default_font

        sx, sy = sxy
        # ----- Draw Information Text ----- #
        image_draw.text(
            xy=(sx + indent_space, sy),
            text=text,
            fill=font_color,
            font=font
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
    def _calc_optimal_grid_size(n_images, n_target_column=-1):
        """
        Calculate optimal grid size.

        Args:
            n_images (int):
            n_target_column (int):

        Returns:
            (int, int):

        """
        if n_target_column == -1:
            n_column = np.ceil(np.sqrt(n_images))
        else:
            n_column = min(n_images, n_target_column)

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

    @staticmethod
    def _gray_scale_to_rgb(image):
        n_channels = image.shape[0]
        if n_channels == 1:
            return np.concatenate([image, image, image], axis=0)
        elif n_channels == 3:
            return image
        else:
            raise ValueError(f"Unexpected GrayScale Image Shape. {image.shape}")

    @staticmethod
    def _mask_to_heatmap(mask):
        """
        Create Class Activation Map.

        Args:
            mask (np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            [C, H, W] -> [3, H, W]

        """
        return Mask.to_jet(mask)

    @staticmethod
    def _mask_to_rgb(mask):
        """
        Colorize prediction mask.

        Args:
            mask (np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            [C, H, W] -> [3, H, W]

        """
        alpha = np.max(mask, axis=0)
        indexes = np.argmax(mask, axis=0)

        rgb_mask = Mask.index_image_to_rgb_image(indexes) * alpha

        return rgb_mask

    @staticmethod
    def _gt_mask_to_rgb(index_mask):
        """
        Colorize Ground-Truth mask.

        Args:
            index_mask (np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            ([H, W] or [1, H, W]) -> [3, H, W]

        """
        return Mask.index_image_to_rgb_image(index_mask)

    @staticmethod
    def _blend(image1, image2, alpha=0.5):
        """
        Blend image1 with image2.

        Args:
            image1 (np.ndarray):
            image2 (np.ndarray):
            alpha (float):

        Returns:
            np.ndarray

        Shapes:
            ([3, H, W], [3, H, W]) -> [3, H, W]

        """
        return (1.0 - alpha) * image1 + alpha * image2
