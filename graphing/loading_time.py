import os
import sys
import time
import glob
from typing import List

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.filenamingtool import FileNamingTool


def get_loading_time(paths: List[str]):

    start_time = time.time()

    for path in paths:

        np.load(path)

    end_time = time.time()

    return end_time - start_time


def get_synthetic_loading_time():

    synthetic = glob.glob("./generated_files/synthetic/*.npz")

    loading_time = get_loading_time(synthetic)

    return loading_time


def get_original_loading_time():

    original = glob.glob("./generated_files/original/*.npz")

    loading_time = get_loading_time(original)

    return loading_time


if __name__ == "__main__":

    arr = []

    for _ in range(1000):
        arr.append(get_original_loading_time())

    narr = np.array(arr)

    df = pd.DataFrame(narr, columns=["loading_time_sec"])

    fname = "original-loading-time"
    source = "eagleimagenet30000"
    fname = FileNamingTool.generate_filename("./results", fname, "csv", source)

    df.to_csv(
        fname,
    )
