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

    synthetic = glob.glob("./generated_files/synthetic/*.npz")[:20000]

    loading_time = get_loading_time(synthetic)

    return loading_time


def get_original_loading_time():

    original = glob.glob("./generated_files/original/*.npz")[:20000]

    loading_time = get_loading_time(original)

    return loading_time


if __name__ == "__main__":

    arr = []
    iter = 0
    for _ in range(5):
        arr.append(get_synthetic_loading_time())

        iter += 1

        print(iter)

    narr = np.array(arr)

    df = pd.DataFrame(narr, columns=["loading_time_sec"])

    fname = "20000-imgs-5-times-synthetic-loading-time"
    source = "3=3e"
    fname = FileNamingTool.generate_filename("./results", fname, "csv", source)

    df.to_csv(
        fname,
    )
