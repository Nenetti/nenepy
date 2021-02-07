from collections import Counter

import torchvision
from torchvision.models.detection.image_list import ImageList

from .value import Value
from .dict_value import DictValue


class Values:

    def __init__(self, values):
        self.values = values
        self.value_strs, self.value_coefficients = self._summarize_texts(values)
        self.n_print_elements = len(self.value_strs)
        # self.texts = self.to_texts(self.value_strs, self.value_coefficients)
        self.text = self._to_text(values)
        # self._type = self._to_type(values)

    @staticmethod
    def _summarize_texts(values):
        texts = []
        for value in values:
            if isinstance(value, (Value, Values, DictValue)):
                texts.append(value.text)
            else:
                raise ValueError

        counter_sorted = Counter(texts).most_common()

        summary_texts = []
        summary_coefficients = []
        for value, n in counter_sorted:
            summary_texts.append(value)
            summary_coefficients.append(n)

        return summary_texts, summary_coefficients

    @staticmethod
    def to_texts(value_strs, value_coefficients):
        texts = []
        for value_str, value_coefficient in zip(value_strs, value_coefficients):
            text = f"{value_str} * {value_coefficient}"
            texts.append(text)

        return texts

    @staticmethod
    def _to_text(values):
        return str([value.text for value in values])

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)
