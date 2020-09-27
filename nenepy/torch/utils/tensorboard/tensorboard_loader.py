import shutil
from enum import Enum, auto
from pathlib import Path
from time import sleep

import cv2
import tensorboard
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from tqdm import tqdm


class Type(Enum):
    SCALAR = auto()
    IMAGE = auto()


class TensorBoardLoader:

    def __init__(self, log_dir, output_dir):
        """

        Args:
            output_dir:
        """
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.recursive(Path(log_dir))

    def load_log(self, log_file):
        event_acc = EventAccumulator(str(log_file), size_guidance={"images": 0})
        event_acc.Reload()
        image_tags = event_acc.Tags()['images']
        for tag in tqdm(image_tags, leave=False, ascii=True):
            tag_name = Path(tag)
            if tag_name.suffix != "":
                tag_name = tag_name.parent.joinpath(tag_name.stem)

            dir_path = self.output_dir.joinpath(tag_name)
            dir_path.mkdir(exist_ok=True, parents=True)
            for index, event in enumerate(event_acc.Images(tag)):
                output_path = dir_path.joinpath(f"{str(index)}.png")
                image = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Image.fromarray(image).save(output_path)

    def recursive(self, dir_path):
        """

        Args:
            dir_path (Path):

        Returns:

        """
        for path in dir_path.iterdir():
            if path.is_file():
                self.load_log(path)
            elif path.is_dir():
                self.recursive(path)
