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
        order: int = 1,
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

        plt.scatter(entropy, cratio, color="blue", alpha=0.5)

        # plt.title(ptitle, fontsize=20, pad=20)
        plt.xlabel(f"Entropy ($H_{order}$)", fontsize=24, labelpad=12)
        plt.ylabel("Compression Ratio", fontsize=24, labelpad=12)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.tight_layout(pad=2)

        if save:
            plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def graph_higher_order_entropy_and_compression_ratio(
        data: str,
        compression_type: str,
        source: str,
        save: bool = False,
        order: int = 1,
        channel: str = "red",
    ):
        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        def theoretical_maximumCr(x):
            return 8 / x

        x = np.array(np.arange(0.1, 8, 0.1))
        y = theoretical_maximumCr(x)

        df = pd.read_csv(data)
        num_rows = df.shape[0]

        fname = "{}-{}-imgs-entropy-and-{}-compression-ratio-plot".format(
            num_rows, source, compression_type
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

        if channel:
            entropy_label = f"E{order}_{channel}"
            xlabel = f"Entropy ($H_{order}$) for {channel} channel" 
        else:
            entropy_label = f"E{order}"
            xlabel = f"Entropy ($H_{order}$)" 

        entropy = df[entropy_label]

        plt.figure(figsize=(12, 8))
        # graph original images data
        plt.scatter(
            entropy,
            cratio,
            alpha=0.5,
        )

        plt.plot(
            x,
            y,
            label=f"Theoretical Maximum Compression Ratio (8/$H_{order}$)",
            color="black",
            linestyle="dashed",
        )
        plt.xlabel(xlabel, fontsize=24, labelpad=12)
        plt.ylabel("Compression Ratio", fontsize=24, labelpad=12)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xlim([0, 8])
        plt.ylim([0, 20])

        plt.legend(loc="upper right", fontsize=20)

        if save:
            plt.savefig(save_fname)

        plt.show()


if __name__ == "__main__":

    paths = {
        "1": "./results/20240430T121325==1=eagleimagenet--results-imagenet-rand-300000.csv",
        "1c": "./results/20240605T052200==1c--300000-rand-processed-images-results.csv",
        "3b": "./results/20240626T021406==3b--30000-e2-redchannel-processed-images-results.csv",
        "3c": "./results/20240626T023210==3c--30000-e3-redchannel-processed-images-results.csv",
    }

    EntropyCompressionRatioPlots.graph_higher_order_entropy_and_compression_ratio(
        paths["3c"], "npz", "3c", save=True, order=3
    )
