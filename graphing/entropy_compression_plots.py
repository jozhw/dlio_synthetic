import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.filenamingtool import FileNamingTool


class EntropyCompressionRatioPlots:

    @staticmethod
    def graph_entropy_and_compression_ratio(
        data: str,
        compression_type: str,
        source: str,
        save: bool = False,
    ):
        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        df = pd.read_csv(data)
        num_rows = df.shape[0]

        fname = "{}-entropy-and-{}-compression-ratio-plot".format(
            num_rows, compression_type
        )
        ptitle = "Entropy and Compression Ratio for {} Images".format(
            num_rows, compression_type
        )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source
        )

        # get the specific name of the compression_ratio, which the data
        # has the first part of the naming be the extension of the compresse
        # file
        cr_name = "{}_compression_ratio".format(compression_type)

        # the o prefix indicates original and the s indicates synthetic

        cratio = df[cr_name]

        entropy = df["entropy"]

        plt.figure(figsize=(12, 8))

        plt.rc('xtick',labelsize=30)
        plt.rc('ytick',labelsize=30)

        plt.scatter(entropy, cratio, color="blue", alpha=0.5)

        #plt.title(ptitle, fontsize=20, pad=20)
        plt.xlabel("Entropy", fontsize=32, labelpad=12)
        plt.ylabel("Compression Ratio", fontsize=32, labelpad=12)

        plt.tight_layout(pad=2)

        if save:
            plt.savefig(save_fname)

        plt.show()


if __name__ == "__main__":

    paths = {
        "1": "./results/20240430T121325==1=eagleimagenet--results-imagenet-rand-300000.csv",
        "1c": "./results/20240605T052200==1c--300000-rand-processed-images-results.csv"
    }

    EntropyCompressionRatioPlots.graph_entropy_and_compression_ratio(
        paths["1"], "npz", "1", save=True
    )
