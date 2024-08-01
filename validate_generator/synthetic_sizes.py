from typing import List

import os
import sys

import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.filenamingtool import FileNamingTool
from utils.saving import Saving


def _get_compressed_size(path):

    return os.path.getsize(path)


def _get_uncompressed_size(path):

    arr = np.load(path)

    # since only one arr which np defines as "arr_0"


    dims = arr["arr_0"].shape

    if len(dims) == 2:
        return dims[0] * dims[1]

    elif len(dims) == 3:
        return dims[0] * dims[1] * dims[2]

    else:
        raise ValueError(f"The dimensions are not valid for: {dims}")

def get_synthetic_sizes(path):

    fname, _ = os.path.splitext(os.path.basename(path))

    # compressed size
    csize = _get_compressed_size(path) 

    ucsize = _get_uncompressed_size(path)

    return {
        "filename": fname,
        "npz_compressed_image_size": csize,
        "uncompressed_size": ucsize
    }


def run(paths: List, source, tosave=False):

    results = []

    for path in paths:

        result = get_synthetic_sizes(path)

        results.append(result)

    if tosave:

        raw_filename = "synthetic-image-sizes"

        save_path = FileNamingTool.generate_filename("./results/", raw_filename, "csv", source)
        Saving.save_to_csv(results, save_path)

    else:

        print(results)

    return 0


    

if __name__ == "__main__":

    import glob

    paths = glob.glob("./generated_files/synthetic/*npz")

    results = run(paths, "3f", tosave=True)
