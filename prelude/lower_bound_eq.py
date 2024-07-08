import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def _inverse_model(x, a, h, k):

    return a / (x - h) + k


def find_inverse_model(x: np.ndarray, y: np.ndarray, model):
    popt, pcov = curve_fit(model, x, y, p0=(8, 0, 0))

    a, h, k = popt

    print(popt)

    x_fit = np.linspace(min(x), max(x), 300)
    y_fit = model(x_fit, *popt)

    plt.scatter(x, y, color="red", label="Completely Random")
    plt.plot(x_fit, y_fit, label=f"Fitted curve: y = {a:.2f} / (x - {h:.2f}) + {k:.2f}")

    plt.xlabel("Entropy")
    plt.ylabel("Compression Ratio")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = "./results/20240605T052200==1c--300000-rand-processed-images-results.csv"
    df = pd.read_csv(path)
    x = np.array(df["entropy"])
    y = np.array(df["npz_compression_ratio"])
    find_inverse_model(x, y, _inverse_model)
