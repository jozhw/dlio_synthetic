import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy, norm

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.filenamingtool import FileNamingTool
from utils.saving import Saving


class PDF:

    @staticmethod
    def _generate_pdf(pdx=np.arange(256), mean: float = 127.0, std: float = 30.0):
        # gen probability density function
        pdf = norm.pdf(pdx, mean, std)

        return pdf

    @staticmethod
    def _pdf_calculate_entropy(pdf):

        # calculate shannon entropy using the probability density function
        shannon_ent = entropy(pdf, base=2)

        return shannon_ent


class EntropyStd:

    @staticmethod
    def _entropy_calculate_std(
        target_entropy,
        lower_bound: float = 0.01,
        upper_bound: float = 5000.0,
        tolerance: float = 1e-6,
    ):
        if target_entropy > 8:
            raise ValueError("RGB image entropy cannot be greater than 8.")

        while abs(upper_bound - lower_bound) > tolerance:
            mid_std = (lower_bound + upper_bound) / 2
            pdf = PDF._generate_pdf(std=mid_std)
            mid_entropy = PDF._pdf_calculate_entropy(pdf)

            if mid_entropy < target_entropy:
                lower_bound = mid_std
            else:
                upper_bound = mid_std

        return (lower_bound + upper_bound) / 2

    @staticmethod
    def entropy_pdf_analyzer(target_entropies: np.ndarray):
        # results array of dictionaries to save to csv
        results: List[Dict] = []

        for target_entropy in target_entropies:
            # local scope result dictionary
            result: Dict = {}

            est_std = EntropyStd._entropy_calculate_std(target_entropy)
            pdf = PDF._generate_pdf(std=est_std)
            est_entropy = PDF._pdf_calculate_entropy(pdf)
            target_est_entropy_diff = target_entropy - est_entropy

            result["target_entropy"] = target_entropy
            result["estimated_std"] = est_std
            result["estimated_entropy"] = est_entropy
            result["target_est_entropy_diff"] = target_est_entropy_diff

            results.append(result)

        fname = FileNamingTool.generate_filename(
            "./results", "results-entropy-to-std", "csv", "local"
        )
        Saving.save_to_csv(results, fname)

    @staticmethod
    def run():
        target_entropies = np.arange(0.1, 8.00, 0.001)
        EntropyStd.entropy_pdf_analyzer(target_entropies)

    @staticmethod
    def graph(path):

        df = pd.read_csv(path)
        ratio = df["target_est_entropy_diff"]
        target_entropy = df["target_entropy"]

        plt.figure(figsize=(12, 8))
        plt.scatter(
            target_entropy, ratio, color="blue", alpha=0.5, marker="o", linestyle="-"
        )
        #plt.title("Target Entropy - Estimated Entropy Difference by Target Entropy")
        plt.xlabel(f"Target Entropy ($H_1$)", fontsize=24, labelpad=12)
        plt.ylabel(f"Entropy ($H_1$) Difference", fontsize=24, labelpad=12)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.grid(True)

        fname = FileNamingTool.generate_filename(
            "./results/plots", "target-entropy-diff-plot", "png", "local"
        )
        plt.savefig(fname)

        plt.show()


class PDFEntropy:

    @staticmethod
    def _calculate_entropies(means: List[int], stds: np.ndarray) -> List[Dict]:
        results = []
        pdx = np.arange(256)
        for mean in means:
            for std in stds:
                pdf = PDF._generate_pdf(pdx, mean, std)
                entropy = PDF._pdf_calculate_entropy(pdf)
                results.append({"mean": mean, "std": std, "entropy": entropy})
        return results

    @staticmethod
    def run():
        means = [127]
        array1 = np.arange(0.01, 2.01, 0.02)
        array2 = np.arange(2, 30.1, 0.1)
        array3 = np.arange(30, 3000.5, 0.5)
        stds = np.concatenate((array1, array2, array3))

        results = PDFEntropy._calculate_entropies(means, stds)

        fname = FileNamingTool.generate_filename(
            "./results", "results-pdf-to-entropy", "csv", "local"
        )

        Saving.save_to_csv(results, fname)

    @staticmethod
    def graph(path):
        df = pd.read_csv(path)

        fname = FileNamingTool.generate_filename(
            "./results/plots", "std-and-entropy-plot", "png", "local"
        )

        ptitle = "Standard Deviation of Gaussian Distribution and Entropy (E1)"

        entropy = df["entropy"]
        std = df["std"]

        plt.figure(figsize=(12, 8))
        plt.scatter(std, entropy, color="blue", alpha=0.5)
        #plt.title(ptitle)
        plt.xlabel("Standard Deviation", fontsize=24, labelpad=12)
        plt.ylabel(f"Entropy ($H_1$)", fontsize=24, labelpad=12)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.xlim([0, 100])

        #plt.grid(True)

        plt.savefig(fname)

        plt.show()


if __name__ == "__main__":
    PDFEntropy.graph("./results/20240626T210249==4--results-pdf-to-entropy.csv")
    #EntropyStd.graph("./results/20240626T210438==5--results-entropy-to-std.csv")
