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


class NPZLoadingErrors:

    def __init__(self, path_to_original_data, path_to_synthetic_data):
        self.odf = pd.read_csv(path_to_original_data)
        self.sdf = pd.read_csv(path_to_synthetic_data)

        self.num_files = len(self.odf["loading_time_sec"])

        if self.num_files != len(self.sdf["loading_time_sec"]):
            raise ValueError("Wrong dataset")

    def individual_loading_error(self):
        results = []
        # since transforming from pd Dataframe to np array preserves indexing, opt to do the following
        oloading = self.odf["loading_time_sec"]
        sloading = self.sdf["loading_time_sec"]

        # since the original is the ground truth the error rate is (synthetic - original) / original

        for i in range(len(oloading)):
            otime = oloading[i]
            stime = sloading[i]

            e = (stime - otime) / otime

            results.append(e)

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

    @staticmethod
    def run():

        original = "./results/20240709T220046==3=3d5--29000-imgs-100-chunk-original-loading-time.csv"
        synthetic = "./results/20240709T220252==3=3d5--29000-imgs-100-chunk-synthetic-loading-time.csv"

        err = NPZLoadingErrors(original, synthetic)
        results = err.individual_loading_error()

        df = pd.DataFrame(results)
        df.to_csv(
            "./results/==3=3d5--synthetic-npz-compressed-loading-error.csv",
            header=["error"],
            index=False,
        )


if __name__ == "__main__":
    NPZLoadingErrors.run()
