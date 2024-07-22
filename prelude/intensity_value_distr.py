import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from calculations import Calculations
from utils.filenamingtool import FileNamingTool

def graph_intensity_distribution(
    path_to_image: str,
    save_path="results/plots/",
):

    # get file name
    raw_fname = os.path.basename(path_to_image)
    fname = os.path.splitext(raw_fname)[0]

    # save path
    save_path = FileNamingTool.generate_filename(save_path, "image-{}-intensity-distr".format(fname), "png", "localcatsanddogs")

    # load image into numpy array
    image: np.ndarray = np.array(Image.open(path_to_image))

    occurances = Calculations.count_occurrences(image)

    # Extract keys and values
    keys = list(range(256))
    values = [occurances.get(key, 0) for key in keys]

    nocc = np.array(values)

    nprob = nocc / nocc.sum()

    plt.figure(figsize=(12, 8))
    plt.bar(keys, nprob, color="blue", alpha=0.7)
    plt.xlabel("Intensity Value", fontsize=28, labelpad=12)
    plt.ylabel("Frequency", fontsize=28, labelpad=12)
    plt.title(
        "Pixel Intensity Value Distribution for Figure 1".format(
            fname, 
        ), fontsize=28, pad=12
    )
    #plt.grid(True)

    plt.tight_layout(pad=2)

    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    paths = [
        "./assets/cats_and_dogs_images/8144.jpg",
        "./assets/cats_and_dogs_images/8411.jpg"
    ]

    graph_intensity_distribution(paths[0])
