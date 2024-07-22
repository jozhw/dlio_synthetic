import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.filenamingtool import FileNamingTool


class LoadingErrorPlots:

    @staticmethod
    def graph_scatter(path_error_data, source, save=False):

        df = pd.read_csv(path_error_data)

        fname = FileNamingTool.generate_filename(
            "./results/plots",
            "synthetic-original-loading-error-scatter-plot",
            "png",
            source,
        )
        plt.figure(figsize=(12, 8))
        plt.scatter(
            df.index,
            df["error"],
            color="blue",
            marker="2",
            s=50,
            label="Loading Error Values",
        )
        plt.axhline(y=0, color="red", linestyle=(0, (1, 10)), label="error = 0")
        plt.axhline(y=0.05, color="red", linestyle="--", label="error = 5%")
        plt.axhline(
            y=-0.05,
            color="red",
            linestyle="--",
        )
        plt.xlabel("Index")
        plt.ylabel("Loading Error")
        plt.title("Lossless NPZ (Deflate) Synthetic Image Loading Time Error", fontsize=14, pad=20)
        plt.legend(loc="lower right")
        if save:
            plt.savefig(fname)
        plt.show()

    @staticmethod
    def graph_histo(path_error_data, source, save=False):

        df = pd.read_csv(path_error_data)

        fname = FileNamingTool.generate_filename(
            "./results/plots",
            "synthetic-original-loading-error-histogram",
            "png",
            source,
        )
        arr = np.array(df["error"]) * 100
        x = np.mean(arr)
        min = np.min(arr)
        max = np.max(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        var_x = np.var(arr)
        std = np.std(arr)

        plt.figure(figsize=(12, 8))
        plt.hist(arr, bins=100, range=(-25, 25), color="red", alpha=0.5)
        plt.axvline(x=0, color="black", linestyle=(0, (1, 10)), label="error = 0%")
        plt.axvline(x=5, color="black", linestyle="dotted", label="error = 5%")
        plt.axvline(x=-5, color="black", linestyle="dotted")
        plt.title("Lossless NPZ (Deflate) Synthetic Image Loading Time Error Percent", fontsize=20, pad=20)
        plt.legend(loc="upper right")
        plt.xlabel("Loading Time Relative Difference (%) with Chunk Size of 100", fontsize=18, labelpad=15)
        plt.ylabel("Occurance", fontsize=18, labelpad=15)

        # Create a string with the statistical values
        stats_str = f"Mean: {x:.4f}\nStd Dev: {std:.4f}\nVariance: {var_x:.4f}\nMinimum: {min:.4f}\nMaximum: {max:.4f}\n25th Percentile: {q1:.4f}\n75th Percentile: {q3:.4f}"

        # Add the description box with statistical values
        plt.text(
            0.05,
            0.95,
            stats_str,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="square", facecolor="wheat", alpha=0.5),
        )

        if save:
            plt.savefig(fname)

        plt.show()


if __name__ == "__main__":
    LoadingErrorPlots.graph_histo("./results/20240710T231232==3=3d5--synthetic-npz-compressed-loading-error.csv", "3=3d5",save=True)
