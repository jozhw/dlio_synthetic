import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.removing import Removing


# compare the compression ratio calculations for flattened array vs the original dimensions
def compare_flat_and_original(path):
    filename, _ = os.path.splitext(os.path.basename(path))

    # set the values within the numpy array to uint8
    # may change this sometime in the future
    image: np.ndarray = np.array(Image.open(path)).astype(np.uint8)

    # numpy reads the image in the order of Height, Width, Depth
    dimensions: Tuple = image.shape

    total_pixels = dimensions[0] * dimensions[1]

    size_bytes = total_pixels * dimensions[2]

    original_cpath = Path("./generated_files/{}.npz".format(filename))

    flattened_cpath = Path("./generated_files/flattened_{}.npz".format(filename))

    np.savez_compressed(original_cpath, image)

    flattened_image = image.flatten()

    np.savez_compressed(flattened_cpath, flattened_image)

    # calculate compression ratios

    original_csize = os.path.getsize(original_cpath)

    flattened_csize = os.path.getsize(flattened_cpath)

    original_cratio = size_bytes / original_csize

    flattened_cratio = size_bytes / flattened_csize

    original_flattened_cratio = original_cratio / flattened_cratio

    result = {
        "filename": filename,
        "original_csize": original_csize,
        "flattened_csize": flattened_csize,
        "original_cratio": original_cratio,
        "flattened_cratio": flattened_cratio,
        "original_flattened_cratio": original_flattened_cratio,
    }

    Removing.remove_compressed_imgs(original_cpath)
    Removing.remove_compressed_imgs(flattened_cpath)

    return result


if __name__ == "__main__":

    paths = [
        "./assets/test_images/test1.jpg",
        "./assets/test_images/test2.jpg",
        "./assets/test_images/test3.jpg",
        "./assets/test_images/test4.jpg",
        "./assets/test_images/test5.jpg",
    ]
    # i want the compression ratio for the adjusted dimensions and the original

    for path in paths:
        print(compare_flat_and_original(path))
