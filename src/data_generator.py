import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from scipy.stats import entropy, norm

from compressions import Compressions
from utils.filenamingtool import FileNamingTool
from utils.saving import Saving
from utils.validations import Validations


class DeflateDataGenerator:

    @staticmethod
    def _calculate_entropy(compression_ratio):

        entropy = 5.59 / (compression_ratio - 0.32) + 0.06

        return entropy

    @staticmethod
    def _generate_pdf(pdx=np.arange(256), mean: float = 127.0, std: float = 30.0):
        # gen probability density function
        pdf = norm.pdf(pdx, mean, std)
        pdf /= pdf.sum()  # normalize the probabilities to sum to 1
        return pdf

    @staticmethod
    def _pdf_calculate_entropy(pdf):
        # calculate shannon entropy using the probability density function
        shannon_ent = entropy(pdf, base=2)

        return shannon_ent

    @staticmethod
    def _calculate_std(
        target_entropy,
        lower_bound: float = 0.01,
        upper_bound: float = 5000.0,
        tolerance: float = 1e-6,
    ):
        if math.floor(target_entropy) > 8:
            raise ValueError("RGB image entropy cannot be greater than 8.")

        while abs(upper_bound - lower_bound) > tolerance:
            mid_std = (lower_bound + upper_bound) / 2
            pdf = DeflateDataGenerator._generate_pdf(std=mid_std)
            mid_entropy = DeflateDataGenerator._pdf_calculate_entropy(pdf)

            if mid_entropy < target_entropy:
                lower_bound = mid_std
            else:
                upper_bound = mid_std

        return (lower_bound + upper_bound) / 2

    @staticmethod
    def _generate_intensity_values(std, size: int) -> np.ndarray:
        """
        Generate a NumPy array of the given size with random integer values from a normal distribution
        with the specified mean and standard deviation, within the range [0, 255].
        """
        # calculate the probabilities for each value in the range [0, 255]
        pdx = np.arange(0, 256)

        pdf = DeflateDataGenerator._generate_pdf(std=std)

        values = np.random.choice(pdx, size=size, p=pdf)

        return values.astype(np.uint8)

    @staticmethod
    def generate_synthetic_image(compression_ratio, xside):

        entropy = DeflateDataGenerator._calculate_entropy(compression_ratio)

        std = DeflateDataGenerator._calculate_std(entropy)

        # will create a square image
        size = (xside**2) * 3
        raw_synthetic_image = DeflateDataGenerator._generate_intensity_values(std, size)

        # assume three channels
        synthetic_image = raw_synthetic_image.reshape((xside, xside, 3))

        return synthetic_image


class DataGenerator_v1:
    ACCEPTED_COMPRESSION_TYPES = ["npz", "jpg"]

    def __init__(self, data_path, compression_types, source):

        self.data_path = data_path
        self.compression_types = compression_types
        self.source = source

    def _validations(self):
        Validations.validate_compression_types(
            DataGenerator_v1.ACCEPTED_COMPRESSION_TYPES, self.compression_types
        )

    def _load_data(self):
        df = pd.read_csv(self.data_path)

        return df

    def generate_deflate(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            df = self._load_data()
        else:
            df = None

        df = comm.bcast(df, root=0)

        num_rows = len(df)
        rows_per_process = num_rows // size
        start = rank * rows_per_process
        end = start + rows_per_process if rank != size - 1 else num_rows

        results: List[Dict[str, Any]] = []
        for _, row in df.iloc[start:end].iterrows():
            file_name: str = row.at["file_name"]
            original_size = row.at["uncompressed_size"]
            ocratio = row.at["npz_compression_ratio"]
            xside = int(np.sqrt(original_size / 3))
            sdimensions = (xside, xside, 3)
            ssize = (xside**2) * 3

            # synthetics
            sfile_name: str = "synthetic-" + str(file_name)

            synthetic_image = DeflateDataGenerator.generate_synthetic_image(
                ocratio, xside
            )

            result = Compressions.compress_and_calculate(
                sfile_name, synthetic_image, sdimensions, ["npz"], self.source
            )

            result["uncompressed_size"] = ssize
            result["file_name"] = sfile_name

            # filter df_images for the current image

            filtered_df: pd.DataFrame = df.loc[df["file_name"] == file_name]
            if filtered_df.empty:
                raise ValueError(f"The file {file_name} does not exist.")

            # ratio between syn/orig compressed size

            results.append(result)

        gathered_results = comm.gather(results, root=0)

        if rank == 0 and gathered_results is not None:

            # get number of images to be processed
            num_rows = len(df)

            raw_filename = "{}-processed-synthetic-images-results".format(num_rows)

            filepath = FileNamingTool.generate_filename(
                "./results/", raw_filename, "csv", self.source
            )

            # merge results from all processes
            flat_results = [
                result for sublist in gathered_results for result in sublist
            ]

            # save to csv here

            Saving.save_to_csv(flat_results, filepath)


############################ testing ###################################333


def test_run():

    test_path = "./results/20240603T170344==localtest--5-processed-images-results.csv"

    import pandas as pd

    df = pd.read_csv(test_path)
    df = df.reset_index()  # make sure indexes pair with number of rows

    results = []

    for index, row in df.iterrows():
        osize = row["uncompressed_size"]
        xside = int(np.sqrt(osize / 3))
        dimensions = (xside, xside, 3)
        fname = "synthetic-{}".format(row["file_name"])
        cratio = row["npz_compression_ratio"]

        synthetic_image = DeflateDataGenerator.generate_synthetic_image(cratio, xside)

        result = Compressions.compress_and_calculate(
            fname, synthetic_image, dimensions, ["npz"], "localtest"
        )
        result["file_name"] = fname

        diff = result["npz_compression_ratio"] - cratio

        diff_ratio = diff / cratio

        print("{}%".format(diff_ratio * 100))

        results.append(result)

    print(results)


if __name__ == "__main__":

    test_path = (
        "./results/20240626T192336==3=eagleimagenet--30000-processed-images-results.csv"
    )
    compression_types = ["npz"]
    source = "eagleimagenet"
    dgen = DataGenerator_v1(test_path, compression_types, source)
    dgen.generate_deflate()
