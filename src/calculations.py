import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class Calculations:

    @staticmethod
    def count_occurrences(image_data: np.ndarray) -> Dict:
        occurrences: Dict = {}
        for row in image_data:
            for pixel in row:
                for value in pixel:
                    if value in occurrences:
                        occurrences[value] += 1
                    else:
                        occurrences[value] = 1
        return occurrences

    @staticmethod
    def calculate_entropy(occurrences: Dict) -> float:

        # the summation of all the occurrences for each intensity value will
        # give you the image size
        file_size: int = 0
        for value in occurrences.values():
            file_size += value

        entropy: float = 0
        for count in occurrences.values():
            if count != 0:
                entropy += -(count / file_size) * math.log2(count / file_size)
        return entropy

    @staticmethod
    def calculate_compression_ratio(compressed_filepath: Path, dimensions):
        # Get the size of the image file
        compressed_file_size: int = os.path.getsize(compressed_filepath)

        # Calculate the number of pixels in the image
        num_pixels: int = dimensions[0] * dimensions[1]

        num_dimensions: int = len(dimensions)

        # if three dimensions, know that there are more than one channel, non black and white image
        if num_dimensions == 3:
            num_channels: int = dimensions[2]
        # if there are two dimensions, know that it is a black and white image so set channels to 1
        elif num_dimensions == 2:
            num_channels: int = 1
        else:
            raise ValueError(
                "Invalid dimensions. Valid Dimensions are length 2 or 3, received {}".format(
                    num_dimensions
                )
            )

        # Calculate the compression ratio
        # must be greater than or equal to 1
        # equation is uncompressed/compressed
        compression_ratio: float = (num_pixels * num_channels) / compressed_file_size

        return compression_ratio

    @staticmethod
    def calculate_mean_intensity_value(occurrences: Dict) -> int:

        total_sum: int = 0
        total_count: int = 0

        for key, value in occurrences.items():

            total_sum += key * value
            total_count += value

            if total_count == 0:
                return 0

        # round mean
        mean: int = round(total_sum / total_count)
        return mean

    @staticmethod
    def calculate_squared_equivalent(dimensions) -> int:

        a = dimensions[0] * dimensions[1]

        rounded_square = int(math.sqrt(a))

        return rounded_square
