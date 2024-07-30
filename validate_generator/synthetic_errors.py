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


class NPZErrors:

    def __init__(self, path_to_original_data, path_to_synthetic_data):
        self.odf = pd.read_csv(path_to_original_data)
        self.sdf = pd.read_csv(path_to_synthetic_data)

        self.num_files = len(self.odf["npz_compressed_image_size"])

        if self.num_files != len(self.sdf["npz_compressed_image_size"]):
            raise ValueError("Wrong dataset")

    def npz_individual_compression_ratio_error(self):
        results = []
        for _, row in self.odf.iterrows():
            file_name = row.at["file_name"]
            ocratio = row.at["npz_compression_ratio"]

            # find corresponding synthetic
            sfile_name = f"synthetic-{file_name}"
            scratio = self.sdf.loc[
                self.sdf["file_name"] == sfile_name, "npz_compression_ratio"
            ].values[0]

            error = (scratio - ocratio) / ocratio

            results.append(error)

        results = np.array(results)
        mean = results.mean()
        std_dev = results.std()
        minimum = results.min()
        maximum = results.max()
        q1 = np.percentile(results, 25)
        q3 = np.percentile(results, 75)

        print(f"The error mean is: {mean}")
        print(f"The error std_dev is: {std_dev}")
        print(f"The error minimum is: {minimum}")
        print(f"The error maximum is: {maximum}")
        print(f"The error 25th percentile is: {q1}")
        print(f"The error 75th percentile is: {q3}")

        return results

    def npz_compressed_sums_error(self):

        sum_ocompressed_size = sum(self.odf["npz_compressed_image_size"])
        sum_ouncompressed_size= sum(self.odf["uncompressed_size"])
        sum_scompressed_size = sum(self.sdf["npz_compressed_image_size"])
        sum_suncompressed_size= sum(self.sdf["uncompressed_size"])

        udiff = sum_suncompressed_size - sum_ouncompressed_size

        uerror = udiff / sum_ouncompressed_size * 100

        diff = sum_scompressed_size - sum_ocompressed_size

        error = diff / sum_ocompressed_size * 100
        print(f"The original compressed dataset size is: {sum_ocompressed_size}")
        print(f"The synthetic compressed dataset size is: {sum_scompressed_size}")
        print(
            f"The difference between the synthetic and original compressed dataset size is: {diff}"
        )
        print(f"The error percent is: {error}")

        results = {
            "total_original_uncompressed_size": sum_ouncompressed_size,
            "total_synthetic_uncompressed_size": sum_suncompressed_size,
            "uncompressed_size_difference": udiff,
            "uncompressed_size_error_percent": uerror,
            "total_original_compressed_size": sum_ocompressed_size,
            "total_synthetic_compressed_size": sum_scompressed_size,
            "compressed_size_difference": diff,
            "compressed_size_error_percent": error,
            "number_of_files_each": self.num_files, 
            
        }
        return results

    @staticmethod
    def run():

        original = "./results/20240626T192336==3=eagleimagenet--30000-processed-images-results.csv"
        synthetic = "./results/20240626T224056==3d--29524-processed-synthetic-images-results.csv"

        err = NPZErrors(original, synthetic)
        err.npz_compressed_sums_error()
        results = err.npz_individual_compression_ratio_error()

        df = pd.DataFrame(results)
        df.to_csv(
            "./results/synthetic-compression-ratio-error.csv",
            header=["error"],
            index=False,
        )

    @staticmethod
    def run_exact_distr():
        
        original = "./results/20240626T192336==3=eagleimagenet--30000-processed-images-results.csv"
        synthetic = "./results/20240723T011532==3=3e--29524-processed-synthetic-images-results.csv"

        err = NPZErrors(original, synthetic)
        results = err.npz_compressed_sums_error()

        print(results)

        #df = pd.DataFrame(results)
        #df.to_csv(
        #    "./results/synthetic-compression-ratio-error.csv",
        #    header=["error"],
        #    index=False,
        #)

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
        plt.savefig(fname)
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
        x = np.mean(arr)
        min = np.min(arr)
        max = np.max(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        var_x = np.var(arr)
        std = np.std(arr)

        plt.figure(figsize=(12, 8))
        plt.hist(arr, bins=100, range=(-0.1, 0.1), color="red", alpha=0.5)
        plt.axvline(x=0, color="black", linestyle=(0, (1, 10)), label="error = 0%")
        plt.axvline(x=0.05, color="black", linestyle="dotted", label="error = 5%")
        plt.axvline(x=-0.05, color="black", linestyle="dotted")
        plt.title("Lossless NPZ (Deflate) Compression Ratio Error")
        plt.legend(loc="upper right")
        plt.xlabel("Compression Ratio Error")
        plt.ylabel("Occurance")

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

        plt.savefig(fname)
        plt.show()


if __name__ == "__main__":
    NPZErrors.run_exact_distr()
    #NPZErrors.graph_scatter("./results/synthetic-compression-ratio-error.csv")
