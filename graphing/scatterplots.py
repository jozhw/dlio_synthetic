import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from tools.filenamingtool import FileNamingTool


class ScatterPlots:
    """

    Args:
        paths: with the source as the key (polaris, local, etc.), and the value
        of the source key is also a key, which is the type of compression
        (npz, jpg, none)
    """

    def __init__(self, paths: Dict[str, List[str]]):
        self.paths = paths

    @staticmethod
    def _graph_entropy_and_compression_ratio(
        path: str, source: str, compression_type: str, synthetic: bool = True
    ):

        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        df = pd.read_csv(path)
        num_rows = df.shape[0]

        if synthetic:
            fname = (
                "{}-synthetic-imgs_entropy-and-{}-compression-ratio-plot.png".format(
                    num_rows, compression_type
                )
            )
            ptitle = (
                "Synthetic Image Entropy and {} Compression Ratio for {} Images".format(
                    num_rows, compression_type
                )
            )

        else:
            fname = "{}-imgs-entropy-and-compression-ratio-plot.png".format(num-rows)

            ptitle = "Image Entropy and {} Compression Ratio for {} Images".format(
                num_rows, compression_type
            )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source
        )

        # get the specific name of the compression_ratio, which the data
        # has the first part of the naming be the extension of the compresse
        # file
        cr_name = "{}_compression_ratio".format(compression_type)
        compression_ratio = df[cr_name]

        entropy = df["entropy"]

        plt.figure(figsize=(12, 8))
        plt.scatter(entropy, compression_ratio, color="blue", alpha=0.5)
        plt.title(ptitle)
        plt.xlabel("Entropy")
        plt.ylabel("Compression Ratio")
        plt.grid(True)

        plt.savefig(save_fname)

        plt.show()

    def graph_entropy_and_compression_ratio(self, compression_type: str):

        for source, paths in self.paths.items():

            for path in paths:

                ScatterPlots._graph_entropy_and_compression_ratio(
                    path, source, compression_type=compression_type
                )

    @staticmethod
    def graph_original_and_synthetic_combined_entropy_and_compression_ratio(
        original_data: str, synthetic_data: str, compression_type: str, source: str
    ):
        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        odf = pd.read_csv(original_data)
        sdf = pd.read_csv(synthetic_data)
        num_rows = odf.shape[0]

        fname = "{}-original-and-synthetic-imgs-entropy-and-{}-compression-ratio-plot.png".format(
            num_rows, compression_type
        )
        ptitle = "Original and Synthetic Image Entropy and Compression Ratio for {} Images".format(
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

        ocratio = odf[cr_name]
        scratio = sdf[cr_name]

        oentropy = odf["entropy"]
        sentropy = sdf["entropy"]

        plt.figure(figsize=(12, 8))
        # graph original images data
        plt.scatter(oentropy, ocratio, color="blue", alpha=0.5, label="Original Images")
        # graph synthetic images data
        plt.scatter(sentropy, scratio, color="red", alpha=0.5, label="Synthetic Images")

        plt.title(ptitle)
        plt.xlabel("Entropy")
        plt.ylabel("Compression Ratio")

        plt.xlim([5, 7])
        plt.ylim([0, 5])

        plt.grid(True)

        plt.savefig(save_fname)

        plt.show()


if __name__ == "__main__":
    paths = {
        "polaris": [
            "./results/20240527T191348==1a=polaris--30000-synthetic-imgs-results__jpg_npz.csv"
        ]
    }

    # toGraph = ScatterPlots(paths)
    # toGraph.graph_entropy_and_compression_ratio("npz")

    paths_combined = [
        "./results/20240430T121325==1=polaris--results-imagenet-rand-300000.csv",
        "./results/20240527T191348==1a=polaris--30000-synthetic-imgs-results__jpg_npz.csv",
    ]

    ScatterPlots.graph_original_and_synthetic_combined_entropy_and_compression_ratio(
        paths_combined[0], paths_combined[1], "npz", "polaris"
    )
