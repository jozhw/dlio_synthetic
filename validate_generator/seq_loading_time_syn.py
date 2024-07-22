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


def _get_loading_time(paths: List[str]):

    start_time = time.time()

    for path in paths:

        np.load(path)

    end_time = time.time()

    return end_time - start_time

def iter_get_loading_time(arr: List[List[str]]) -> List[float]:

    loading_times: List[float] = []

    for a in arr:
        t = _get_loading_time(a)

        loading_times.append(t)

    return loading_times

        

def seqchunk_arr(arr: List[str]) -> List[List[str]]:

    chunked_arr = []

    i: int = 2
    n: int = 0
    m: int = 1

    while m <= len(arr) - 1:

        a: List[str] = arr[n:m]

        chunked_arr.append(a)

        n = m
        m += i
        i += 1

    return chunked_arr

    

def get_synthetic_loading_time(arr_length: int) -> List[float]:

    synthetic: List[str] = glob.glob("./generated_files/synthetic/*.npz")[:arr_length]

    chunked_arr: List[List[str]] = seqchunk_arr(synthetic)

    loading_times: List[float] = iter_get_loading_time(chunked_arr)


    return loading_times


def get_original_loading_time(arr_length: int) -> List[float]:

    original: List[str] = glob.glob("./generated_files/original/*.npz")[:arr_length]

    chunked_arr: List[List[str]] = seqchunk_arr(original)

    loading_times: List[float] = iter_get_loading_time(chunked_arr)


    return loading_times


if __name__ == "__main__":

    arr_length = 29000

    arr = get_synthetic_loading_time(arr_length)

    narr = np.array(arr)

    df = pd.DataFrame(narr, columns=["loading_time_sec"])

    fname = "{}-imgs-seqchunk-synthetic-loading-time".format(arr_length)
    source = "2=2n"
    fname = FileNamingTool.generate_filename("./results", fname, "csv", source)

    df.to_csv(
        fname,
    )
