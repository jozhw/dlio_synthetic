import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.filenamingtool import FileNamingTool

class Histograms:

    @staticmethod
    def log_normal_distr(mean, std, size, lower_bound = 1,):
        adjusted_mean = mean - lower_bound
        var = std ** 2
        mu = np.log(adjusted_mean ** 2 / np.sqrt(var + adjusted_mean ** 2))
        sigma = np.sqrt(np.log(var / adjusted_mean ** 2 + 1))

        random_numbers = np.random.lognormal(mean=mu, sigma=sigma, size=size)

        random_numbers += lower_bound
        
        generated_mean = np.mean(random_numbers)
        generated_std = np.std(random_numbers)

        print(random_numbers.sum())

        # plt.figure(figsize=(12, 8))
        # plt.hist(random_numbers, range=(1,5), alpha=0.5,  bins=200)

        return generated_mean, generated_std, random_numbers

    @staticmethod
    def gamma_distr(mean, std, size, lower_bound=1):
        mean = mean - lower_bound
        theta = (std ** 2) / mean
        k = mean / theta
        random_numbers = gamma.rvs(k, scale=theta, size=size)

        random_numbers += lower_bound
        generated_mean = np.mean(random_numbers)
        generated_std = np.std(random_numbers)

        # plt.figure(figsize=(12, 8))
        # plt.hist(random_numbers, range=(1,5), alpha=0.5,  bins=200)

        return generated_mean, generated_std, random_numbers

    @staticmethod
    def graph_compression_ratio(path):

        df = pd.read_csv(path)

        total_size = df["uncompressed_size"].sum()
        compressed_size = df["npz_compressed_image_size"].sum()
        dataset_compression_ratio = total_size / compressed_size

        fname = FileNamingTool.generate_filename(
            "./results/plots",
            "300000-original-and-lognormal-generated-compression-ratio-histogram",
            "png",
            "eagleimagenet",
        )
        arr = np.array(df["npz_compression_ratio"])
        size = len(arr)
        x = np.mean(arr)
        min = np.min(arr)
        max = np.max(arr)
        med = np.median(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        var_x = np.var(arr)
        std = np.std(arr)

        gen_mean, gen_std, random_numbers = Histograms.log_normal_distr(dataset_compression_ratio, std, size)

        print(gen_mean, gen_std)


        plt.figure(figsize=(12, 8))
        plt.hist(random_numbers, range=(1,5), alpha=0.5,  bins=200, label="synthetic compression ratios")
        plt.hist(arr, bins=200, range=(1, 5), color="red", alpha=0.5, label="original compression ratios")
        plt.title("Lossless NPZ (Deflate) Compression Ratio Histogram")
        plt.xlabel("NPZ Compression Ratio (Deflate)")
        plt.ylabel("Occurance")
        plt.axvline(x=dataset_compression_ratio, color="black", linestyle="dotted", label=f"Original Dataset's compression ratio: {dataset_compression_ratio:.4f}")
        plt.legend(loc="lower right")

        # Create a string with the statistical values
        stats_str = f"Original Statistics\nMean: {x:.4f}\nMedian: {med:.4f}\nStd Dev: {std:.4f}\nVariance: {var_x:.4f}\nMinimum: {min:.4f}\nMaximum: {max:.4f}\n25th Percentile: {q1:.4f}\n75th Percentile: {q3:.4f}"

        # Add the description box with statistical values
        plt.text(
            0.75,
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

    Histograms.graph_compression_ratio("./results/20240430T121325==1=eagleimagenet--results-imagenet-rand-300000.csv")
