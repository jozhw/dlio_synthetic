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
            fname = "{}-imgs-entropy-and-compression-ratio-plot.png".format(num_rows)

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

    def graph_multiple_separate_entropy_and_compression_ratio(
        self, compression_type: str
    ):

        for source, paths in self.paths.items():

            for path in paths:

                EntropyCompressionRatioPlots._graph_entropy_and_compression_ratio(
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

    @staticmethod
    def graph_data_comparison_entropy_and_compression_ratio(
        func,
        data1: str,
        source1: str,
        label1: str,
        entropy_label: str,
        label_color: str,
        data2: str,
        source2: str,
        label2: str,
        entropy_label2: str,
        label_color2: str,
        compression_type: str,
    ):
        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        x = np.array(np.arange(0.1, 8, 0.1))
        y = func(x)

        df = pd.read_csv(data1)
        df2 = pd.read_csv(data2)
        num_rows = df.shape[0] + df2.shape[0]

        fname = "{}-{}-and-{}-imgs-entropy-and-{}-compression-ratio-plot".format(
            num_rows, source1, source2, compression_type
        )
        ptitle = "{} and {} Image Entropy and Compression Ratio for {} Images".format(
            source1, source2, num_rows, compression_type
        )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source1
        )

        # get the specific name of the compression_ratio, which the data
        # has the first part of the naming be the extension of the compresse
        # file
        cr_name = "{}_compression_ratio".format(compression_type)

        # the o prefix indicates original and the s indicates synthetic

        cratio = df[cr_name]
        cratio2 = df2[cr_name]

        entropy = df[entropy_label]
        entropy2 = df2[entropy_label2]

        plt.figure(figsize=(16, 9))
        # graph original images data
        plt.scatter(
            entropy,
            cratio,
            color=label_color,
            alpha=0.5,
            label=label1 + " {} Images".format(df.shape[0]),
        )
        # graph synthetic images data
        plt.scatter(
            entropy2,
            cratio2,
            color=label_color2,
            alpha=0.5,
            label=label2 + " {} Images".format(df2.shape[0]),
        )

        plt.plot(
            x,
            y,
            label="Theoretical (8/En)",
            color="black",
            linestyle="dashed",
        )
        plt.title(ptitle)
        plt.xlabel("Entropy")
        plt.ylabel("Compression Ratio")

        plt.xlim([0, 8])
        plt.ylim([0, 20])

        plt.legend(loc="upper right")
        plt.grid(True)

        plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def graph_multi_comparison_entropy_and_compression_ratio(
        func,
        data1: str,
        source1: str,
        label1: str,
        entropy_label: str,
        label_color: str,
        data2: str,
        source2: str,
        label2: str,
        entropy_label2: str,
        label_color2: str,
        data3: str,
        source3: str,
        label3: str,
        entropy_label3: str,
        label_color3: str,
        compression_type: str,
    ):
        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        x = np.array(np.arange(0.1, 8, 0.1))
        y = func(x)

        df = pd.read_csv(data1)
        df2 = pd.read_csv(data2)
        df3 = pd.read_csv(data3)
        num_rows = df.shape[0] + df2.shape[0] + df3.shape[0]

        fname = "{}-{}-and-{}-and{}-imgs-entropy-and-{}-compression-ratio-plot".format(
            num_rows, source1, source2, source3, compression_type
        )
        ptitle = (
            "{} and {} and {} Image Entropy and Compression Ratio for {} Images".format(
                source1, source2, source3, num_rows, compression_type
            )
        )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source1
        )

        # get the specific name of the compression_ratio, which the data
        # has the first part of the naming be the extension of the compresse
        # file
        cr_name = "{}_compression_ratio".format(compression_type)

        # the o prefix indicates original and the s indicates synthetic

        cratio = df[cr_name]
        cratio2 = df2[cr_name]
        cratio3 = df3[cr_name]

        entropy = df[entropy_label]
        entropy2 = df2[entropy_label2]
        entropy3 = df3[entropy_label3]

        plt.figure(figsize=(16, 9))
        # graph original images data
        plt.scatter(
            entropy,
            cratio,
            color=label_color,
            s=30,
            marker=".",
            alpha=0.3,
            label=label1 + " {} Images".format(df.shape[0]),
        )
        # graph synthetic images data
        plt.scatter(
            entropy2,
            cratio2,
            color=label_color2,
            marker=".",
            s=30,
            alpha=0.3,
            label=label2 + " {} Images".format(df2.shape[0]),
        )

        # graph synthetic images data
        plt.scatter(
            entropy3,
            cratio3,
            s=30,
            marker=".",
            color=label_color3,
            alpha=0.3,
            label=label3 + " {} Images".format(df3.shape[0]),
        )
        plt.plot(
            x,
            y,
            label="Theoretical (8/En)",
            color="black",
            linestyle="dashed",
        )
        plt.title(ptitle)
        plt.xlabel("Entropy")
        plt.ylabel("Compression Ratio")

        plt.xlim([0, 8])
        plt.ylim([0, 20])

        plt.legend(loc="upper right")
        plt.grid(True)

        plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def graph_entropy_and_compression_ratio(
        data: str,
        data_name: str,
        compression_type: str,
        source: str,
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

        # graph images data
        plt.scatter(entropy, cratio, color="blue", alpha=0.5, label=data_name)
        # graph synthetic images data

        plt.title(ptitle)
        plt.xlabel("Entropy")
        plt.ylabel("Compression Ratio")
        plt.legend(loc="upper left")
        # plt.xlim([0, 3])
        # plt.ylim([0, 20])
        plt.grid(True)

        plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def theoretical_maximumCr(x):
        return 8 / x

    @staticmethod
    def graph_maximum_scatterplot(
        func, entropy_label, data, data_name, compression_type, source
    ):

        x = np.array(np.arange(0.1, 8, 0.1))
        y = func(x)

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

        entropy = df[entropy_label]

        plt.figure(figsize=(12, 8))

        # graph images data
        plt.scatter(entropy, cratio, color="blue", alpha=0.5, label=data_name)
        # graph synthetic images data
        plt.plot(
            x,
            y,
            label="Theoretical (8/{})".format(entropy_label),
            color="red",
            linestyle="dashed",
        )

        plt.title(ptitle)
        plt.xlabel("Entropy ({})".format(entropy_label))
        plt.ylabel("Compression Ratio")
        plt.legend(loc="upper right")
        # plt.xlim([0, 8])
        # plt.ylim([0, 20])
        plt.grid(True)

        plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def graph_original_and_synthetic_compression_ratio(
        original_data: str, synthetic_data: str, compression_type: str, source: str
    ):
        def function(x):
            return x

        x = np.array(np.arange(0.1, 8, 0.1))
        y = function(x)
        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        odf = pd.read_csv(original_data)
        sdf = pd.read_csv(synthetic_data)
        num_rows = odf.shape[0]

        fname = "{}-original-and-synthetic-imgs-entropy-and-{}-compression-ratio-plot.png".format(
            num_rows, compression_type
        )
        ptitle = "Original and Synthetic Lossless (Deflate) Compression Ratios for {} Images".format(
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

        ocr = odf["npz_compression_ratio"]
        scr = sdf["npz_compression_ratio"]

        plt.figure(figsize=(12, 8))
        # graph original images data
        plt.scatter(ocr, scr, color="blue", alpha=0.5)
        # graph synthetic images data

        plt.plot(
            x,
            y,
            label="y=x",
            color="red",
            linestyle="dashed",
        )
        plt.title(ptitle, fontsize=32)
        plt.xlabel("Compression Ratio of Original", fontsize=32, labelpad=12)
        plt.ylabel("Compression Ratio of Synthetic", fontsize=32, labelpad=12)

        plt.xlim([1, 5])
        plt.ylim([1, 5])

        plt.legend(loc="lower right")
        #plt.grid(True)

        #plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def graph_loading_time_comparision_plot(
        original_data: str, synthetic_data: str, compression_type: str, source: str
    ):
        def function(x):
            return x 

        x = np.array(np.arange(0, 1000, 0.1))
        y = function(x)
        #y1 = y + 0.1
        #y2 = y - 0.1

        # if it is a synthetic image, then the naming of the file should
        # have synthetic in it

        odf = pd.read_csv(original_data)
        sdf = pd.read_csv(synthetic_data)

        fname = "original-and-synthetic-imgs-npz-loading-plot".format(
        )
        ptitle = "Original and Synthetic Lossless (Deflate) Loading Time in Seconds for NPZ Images".format(
        )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source
        )

        # get the specific name of the compression_ratio, which the data
        # has the first part of the naming be the extension of the compresse
        # file
       

        # the o prefix indicates original and the s indicates synthetic

        oloading = odf["loading_time_sec"]
        sloading = sdf["loading_time_sec"]


        plt.figure(figsize=(12, 8))
        # graph original images data
        plt.scatter(oloading, sloading, color="blue", alpha=0.5)
        # graph synthetic images data

        plt.plot(
            x,
            y,
            label="y=x",
            color="red",
            linestyle="dashed",
        )
        #plt.plot(
        #    x,
        #    y1,
        #    label="y = x + 0.1",
        #    color="black",
        #    linestyle="dashed",
        #)
        #plt.plot(
        #    x,
        #    y2,
        #    label="y = x - 0.1",
        #    color="black",
        #    linestyle="dashed",
        #)
        plt.title(ptitle, fontsize=14)
        plt.xlabel("Loading of Original NPZ in Seconds")
        plt.ylabel("Loading of Synthetic NPZ in Seconds")

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.legend(loc="lower right")
        plt.grid(True)

        plt.savefig(save_fname)

        plt.show()

if __name__ == "__main__":

    comparison_paths = [
        "./results/20240626T192336==3=eagleimagenet--30000-processed-images-results.csv",
        "./results/20240626T224056==3d--29524-processed-synthetic-images-results.csv",
    ]

    EntropyCompressionRatioPlots.graph_original_and_synthetic_compression_ratio(comparison_paths[0], comparison_paths[1], "npz", "3=3d")
