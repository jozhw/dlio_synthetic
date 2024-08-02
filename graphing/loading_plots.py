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


class LoadingPlots:

    @staticmethod
    def graph_loading_chunk_comparison(
        original_data: str,
        synthetic_data: str,
        total_imgs: int,
        compression_type: str,
        source: str,
        save: bool = False,
    ):

        def function(x):
            return x

        x = np.array(np.arange(0, 1000, 0.1))
        y = function(x)

        odf = pd.read_csv(original_data)
        sdf = pd.read_csv(synthetic_data)

        size = int(total_imgs / len(odf["loading_time_sec"]))

        fname = "{}-original-and-synthetic-{}-compressed-{}-chunk-loading-plot".format(
            total_imgs, compression_type, size
        )

        ptitle = "Original and Synthetic Lossless ({}) Loading Time in Seconds for {} total Images".format(
            compression_type, total_imgs
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
        label = "Chunk size of {}".format(size)

        plt.figure(figsize=(12, 8))

        plt.scatter(oloading, sloading, color="blue", alpha=0.5, label=label)

        plt.plot(
            x,
            y,
            label="y=x",
            color="red",
            linestyle="dashed",
        )
        plt.title(ptitle, fontsize=14, pad=20)
        plt.xlabel("Loading Time of Original (seconds)")
        plt.ylabel("Loading Time of Synthetic (seconds)")

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.legend(loc="lower right")
        plt.grid(True)

        if save:

            plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def graph_loading_chunk_comparison_bar(
        original_data: str,
        synthetic_data: str,
        total_imgs: int,
        compression_type: str,
        source: str,
        save: bool = False,
    ):

        odf = pd.read_csv(original_data)
        oloading = odf["loading_time_sec"]

        sdf = pd.read_csv(synthetic_data)
        sloading = sdf["loading_time_sec"]

        size = len(oloading)

        oload_toplot = []
        sload_toplot = []
        index = []

        i2 = 0

        for i in range(10):

            oload_toplot.append(oloading[i])
            sload_toplot.append(sloading[i])

            i2 += 1
            index.append(i2)


        # serves as the x axis
        i = np.arange(1, size + 1)

        dfl = pd.DataFrame({
            
            "Original Compressed Image": oload_toplot,
            "Synthetic Compressed Image": sload_toplot

        }, index=index)


        if size != len(sloading):
            raise ValueError(
                "The datasets should be the save length. Got {} for the original and {} for the synthetic".format(
                    size, len(sloading)
                )
            )

        fname = "{}-original-and-synthetic-{}-compressed-chunk-loading-bar-plot".format(
            total_imgs, compression_type
        )
        ptitle = "Original and Synthetic Lossless ({}) Loading Time for a total of {} Images".format(
            compression_type, total_imgs
        )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source
        )

        #plt.rc('xtick',labelsize=12)
        #plt.rc('ytick',labelsize=12)
        dfl.plot.bar(rot=0, figsize=(12,8))
        # plt.title(ptitle, fontsize=20, pad=20)
        plt.xlabel("Chunk Index (chunk size = 100)", fontsize=24, labelpad=12)
        plt.ylabel("Loading Time (seconds)", fontsize=24, labelpad=12)
        plt.tight_layout(pad=2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 20})

        if save:
            plt.savefig(save_fname)

        plt.show()
    @staticmethod
    def graph_loading_seqchunk_comparison(
        original_data: str,
        synthetic_data: str,
        total_imgs: int,
        compression_type: str,
        source: str,
        save: bool = False,
    ):

        odf = pd.read_csv(original_data)
        oloading = odf["loading_time_sec"]

        sdf = pd.read_csv(synthetic_data)
        sloading = sdf["loading_time_sec"]

        size = len(oloading)

        # serves as the x axis
        i = np.arange(1, size + 1)

        if size != len(sloading):
            raise ValueError(
                "The datasets should be the save length. Got {} for the original and {} for the synthetic".format(
                    size, len(sloading)
                )
            )

        fname = "{}-original-and-synthetic-{}-compressed-seqchunk-loading-plot".format(
            total_imgs, compression_type
        )
        ptitle = "Original and Synthetic Lossless ({}) Loading Time for a total of {} Images".format(
            compression_type, total_imgs
        )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source
        )

        plt.figure(figsize=(12, 8))
        # graph original images data
        plt.scatter(
            i, oloading, color="blue", alpha=0.5, label="Original compressed files"
        )
        # graph synthetic images data
        plt.scatter(
            i, sloading, color="red", alpha=0.5, label="Synthetic compressed files"
        )

        plt.title(ptitle, fontsize=14, pad=20)
        plt.xlabel("Number of Files per Chunk")
        plt.ylabel("Loading Time per Chunk (seconds)")

        plt.legend(loc="upper left", prop={'size': 20})
        plt.grid(True)

        if save:
            plt.savefig(save_fname)

        plt.show()

    @staticmethod
    def graph_loading_seqchunk_comparison_bar(
        original_data: str,
        synthetic_data: str,
        total_imgs: int,
        compression_type: str,
        source: str,
        chunk_size: int = 25,
        save: bool = False,

    ):

        odf = pd.read_csv(original_data)
        oloading = odf["loading_time_sec"]

        sdf = pd.read_csv(synthetic_data)
        sloading = sdf["loading_time_sec"]

        size = len(oloading)

        oload_toplot = []
        sload_toplot = []
        index = []

        step = chunk_size
        i2 = 0
        for i in range(chunk_size - 1, size, step):

            if i2 == 0:
                oload_toplot.append(oloading[i])
                sload_toplot.append(sloading[i])
            else:
                oload_toplot.append(oloading[i] - oloading[i - step])
                sload_toplot.append(sloading[i] - sloading[i - step])

            i2 += 1
            index.append(i2)


        # serves as the x axis
        i = np.arange(1, size + 1)

        dfl = pd.DataFrame({
            
            "Original Compressed Image": oload_toplot,
            "Synthetic Compressed Image": sload_toplot

        }, index=index)


        if size != len(sloading):
            raise ValueError(
                "The datasets should be the save length. Got {} for the original and {} for the synthetic".format(
                    size, len(sloading)
                )
            )

        fname = "{}-original-and-synthetic-{}-compressed-seqchunk-loading-bar-plot".format(
            total_imgs, compression_type
        )
        ptitle = "Original and Synthetic Lossless ({}) Loading Time for a total of {} Images".format(
            compression_type, total_imgs
        )

        save_fname = FileNamingTool.generate_filename(
            "./results/plots", fname, "png", source
        )

        #plt.rc('xtick',labelsize=12)
        #plt.rc('ytick',labelsize=12)
        dfl.plot.bar(rot=0, figsize=(12,8))
        # plt.title(ptitle, fontsize=20, pad=20)
        plt.xlabel("Chunk Index (chunk size = {})".format(chunk_size), fontsize=24, labelpad=12)
        plt.ylabel("Loading Time (seconds)", fontsize=24, labelpad=12)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout(pad=2)

        if save:
            plt.savefig(save_fname)

        plt.show()



if __name__ == "__main__":

    comparison_paths = {
        "ochunk1": "./results/20240723T013401==3=3f1--29000-imgs-100-chunk-original-loading-time.csv",
        "schunk1": "./results/20240723T013610==3=3f1--29000-imgs-100-chunk-synthetic-loading-time.csv",
        "oseqchunk": "./results/20240723T021905==3=3f2--29000-imgs-seqchunk-original-loading-time.csv",
        "sseqchunk": "./results/20240723T021658==3=3f2--29000-imgs-seqchunk-synthetic-loading-time.csv",
    }
    LoadingPlots.graph_loading_seqchunk_comparison(
        comparison_paths["oseqchunk"],
        comparison_paths["sseqchunk"],
        29000,
        "npz",
        source="3=3f2",
        save=False
    )

    LoadingPlots.graph_loading_chunk_comparison_bar(
        comparison_paths["ochunk1"],
        comparison_paths["schunk1"],
        29000,
        "npz",
        source="3=3f1",
        save=True
    )

    LoadingPlots.graph_loading_seqchunk_comparison_bar(
        comparison_paths["oseqchunk"],
        comparison_paths["sseqchunk"],
        29000,
        "npz",
        source="3=3f2",
        chunk_size=50,
        save=False
    )

