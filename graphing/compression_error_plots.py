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


class CompressionErrorPlots:

    @staticmethod
    def graph_scatter(path_error_data):

        df = pd.read_csv(path_error_data)

        fname = FileNamingTool.generate_filename(
            "./results/plots",
            "synthetic-original-compression-ratio-error-scatter-plot",
            "png",
            "eagleimagenet",
        )
        plt.figure(figsize=(12, 8))
        plt.scatter(
            df.index,
            df["error"],
            color="blue",
            marker="2",
            s=50,
            label="Compression Ratio (Cr) Error Values",
        )
        plt.axhline(y=0, color="red", linestyle=(0, (1, 10)), label="error = 0")
        plt.axhline(y=0.05, color="red", linestyle="--", label="error = 5%")
        plt.axhline(
            y=-0.05,
            color="red",
            linestyle="--",
        )
        plt.xlabel("Index")
        plt.ylabel("Compression Ratio (Cr) Error")
        plt.title("Lossless NPZ (Deflate) Compression Ratio Error")
        plt.legend(loc="lower right")
        #plt.savefig(fname)
        plt.show()

    @staticmethod
    def graph_histo(path_error_data):

        df = pd.read_csv(path_error_data)

        fname = FileNamingTool.generate_filename(
            "./results/plots",
            "synthetic-original-compression-ratio-error-histogram",
            "png",
            "eagleimagenet",
        )
        arr = np.array(df["error"])
        arr = arr * 100
        x = np.mean(arr)
        min = np.min(arr)
        max = np.max(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        var_x = np.var(arr)
        std = np.std(arr)

        plt.figure(figsize=(12, 8))
        plt.hist(arr, bins=100, range=(-10, 10), color="red", alpha=0.5)
        plt.axvline(x=0, color="black", linestyle=(0, (1, 10)), label="error = 0%")
        plt.axvline(x=5, color="black", linestyle="dotted", label="error = 5%")
        plt.axvline(x=-5, color="black", linestyle="dotted")
        #plt.title("Lossless NPZ (Deflate) Compression Ratio Difference Error", pad=12, fontsize=32)
        plt.legend(loc="upper right")
        plt.xlabel("Compression Ratio Difference Percentage", fontsize=32, labelpad=12)
        plt.ylabel("Occurance", fontsize=32, labelpad=12)

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

        plt.tight_layout(pad=2)
        plt.savefig(fname)
        plt.show()


if __name__ == "__main__":
    CompressionErrorPlots.graph_histo("./results/20240626T225644==3d1--synthetic-compression-ratio-error.csv")
