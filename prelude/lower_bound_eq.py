import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from utils.filenamingtool import FileNamingTool

def _inverse_model(x, a, h, k):

    return a / (x - h) + k


def find_inverse_model(x: np.ndarray, y: np.ndarray, model, source, save: bool = False):
    fname = FileNamingTool.generate_filename("./results/plots", "fitted-curve-rand-compression-ratio-entropy-plot", "png", source)
    popt, pcov = curve_fit(model, x, y, p0=(8, 0, 0))

    a, h, k = popt

    print(popt)

    x_fit = np.linspace(min(x), max(x), 300)
    y_fit = model(x_fit, *popt)

    ptitle = "Entropy and Compression Ratio for {} Images".format(len(x))

    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color="blue", label="Images with Completely Random Pixel Values")
    plt.plot(x_fit, y_fit, color="red", label=f"Fitted curve: y = {a:.2f} / (x - {h:.2f}) + {k:.2f}")

    #plt.title(ptitle, fontsize=20, pad=20)
    plt.xlabel("Entropy", fontsize=24, labelpad=12)
    plt.ylabel("Compression Ratio", fontsize=24, labelpad=12)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout(pad=2)
    plt.legend(prop={'size': 20})

    if save:
        plt.savefig(fname)
    plt.show()


if __name__ == "__main__":
    path = "./results/20240605T052200==1c--300000-rand-processed-images-results.csv"
    df = pd.read_csv(path)
    x = np.array(df["entropy"])
    y = np.array(df["npz_compression_ratio"])
    find_inverse_model(x, y, _inverse_model, "1c", save=True)
