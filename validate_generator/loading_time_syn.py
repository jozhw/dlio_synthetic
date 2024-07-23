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

        

def chunk_arr(arr: List[str], chunk_size) -> List[List[str]]:

    chunked_arr = []

    n: int = 0
    m: int = chunk_size

    num_iter: int = int(len(arr) / chunk_size)

    for _ in range(num_iter):

        a: List[str] = arr[n:m]

        chunked_arr.append(a)

        n += chunk_size
        m += chunk_size

    chunked_len = len(chunked_arr)
    if num_iter != chunked_len:
        raise ValueError(f"The length expected of {num_iter} does not equal the length of the outer array of {chunked_len}")

    return chunked_arr

    

def get_synthetic_loading_time(arr_length: int, chunk_size: int) -> List[float]:

    synthetic: List[str] = glob.glob("./generated_files/synthetic/*.npz")[:arr_length]

    chunked_arr: List[List[str]] = chunk_arr(synthetic, chunk_size)

    loading_times: List[float] = iter_get_loading_time(chunked_arr)


    return loading_times


def get_original_loading_time(arr_length: int, chunk_size: int) -> List[float]:

    original: List[str] = glob.glob("./generated_files/original/*.npz")[:arr_length]

    chunked_arr: List[List[str]] = chunk_arr(original, chunk_size)

    loading_times: List[float] = iter_get_loading_time(chunked_arr)


    return loading_times


if __name__ == "__main__":

    arr_length = 29000

    chunk_size = 100

    arr = get_synthetic_loading_time(arr_length, chunk_size)

    narr = np.array(arr)

    df = pd.DataFrame(narr, columns=["loading_time_sec"])

    fname = "{}-imgs-{}-chunk-synthetic-loading-time".format(arr_length, chunk_size)
    source = "3=3f"
    fname = FileNamingTool.generate_filename("./results", fname, "csv", source)

    df.to_csv(
        fname,
    )
