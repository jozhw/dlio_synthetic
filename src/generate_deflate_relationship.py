from typing import List, Tuple
from numpy.typing import NDArray

import os
import math
import csv

import numpy as np
import pandas as pd

from pathlib import Path

from mpi4py import MPI
from scipy.stats import norm, entropy

from utils.filenamingtool import FileNamingTool


class GenerateDeflateRelationship:

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
        lower_bound: float = 0.001,
        upper_bound: float = 5000.0,
        tolerance: float = 1e-6,
    ):
        if math.floor(target_entropy) > 8:
            raise ValueError("RGB image entropy cannot be greater than 8.")

        while abs(upper_bound - lower_bound) > tolerance:
            mid_std = (lower_bound + upper_bound) / 2
            pdf = GenerateDeflateRelationship._generate_pdf(std=mid_std)
            mid_entropy = GenerateDeflateRelationship._pdf_calculate_entropy(pdf)

            if mid_entropy < target_entropy:
                lower_bound = mid_std
            else:
                upper_bound = mid_std

        return (lower_bound + upper_bound) / 2

    @staticmethod
    def _generate_intensity_values(std, size: int) -> NDArray[np.uint8]:
        """
        Generate a NumPy array of the given size with random integer values from a normal distribution
        with the specified mean and standard deviation, within the range [0, 255].
        """
        # calculate the probabilities for each value in the range [0, 255]
        pdx = np.arange(0, 256)

        pdf = GenerateDeflateRelationship._generate_pdf(std=std)

        values = np.random.choice(pdx, size=size, p=pdf)

        return values.astype(np.uint8)


    @staticmethod
    def npz_deflate(n:int=10, increment: float = 0.001, dim:Tuple[int, int, int]=(224, 224, 3), max_entropy: int = 8):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:

            # initial values
            image_size: int = dim[0] * dim[1] * dim[2]

            # the length of the array of tuples
            index_end: int = int(max_entropy / increment)

            # resolves the inherent floating point issue that is embedded in python by using the numpy datatypes
            entropies: NDArray[np.float16] = np.arange(0, index_end, 1, dtype=np.float16) * increment 

            stds: NDArray[np.float16] = np.zeros(index_end).astype(np.float16)

            # calculate the equivalent standard deviation and add to the stds array
            for i in range(index_end):

                entropy: np.float16 = np.float16(GenerateDeflateRelationship._calculate_std(entropies[i]))

                stds[i] = entropy
        else:

            # ignore this error in lsp due to assignment of None
            entropies, stds, image_size, index_end = None, None, None, None

        # broadcast to all of the workers
        image_size = comm.bcast(image_size, root=0)

        index_end = comm.bcast(index_end, root=0)

        entropies = comm.bcast(entropies, root=0)

        stds = comm.bcast(stds, root=0)

        rows_per_process = index_end // size

        start = rank * rows_per_process
        end = start + rows_per_process if rank != size - 1 else index_end

        # store the results
        # in the order of (index, entropy, std, size, compressed_size, compression_ratio)
        results: List[Tuple[str, np.float16, np.float16, int, int, np.float16]] = []

        # generate the pixel values
        for i in range(start, end):

            for j in range(n):

                std: np.float16 = stds[i]
                entropy: np.float16 = entropies[i]
                index = str(i) + f"-{j}"

                image: NDArray[np.uint8] = GenerateDeflateRelationship._generate_intensity_values(std, image_size)

                fname: str = "./generated_files/{}.npz".format(index)

                # compress to npz files and get the compression ratio
                np.savez_compressed(fname, image)

                compressed_size: int = os.path.getsize(fname)

                # delete the file
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    print(f"File {fname} does not exist.")
                except PermissionError:
                    print(f"Permission denied to delete {fname}.")
                except Exception as e:
                    print(f"Error: {e}")

                compression_ratio: np.float16 = np.float16(image_size / compressed_size)

                results.append((index, entropy, std, size, compressed_size, compression_ratio))


        # gather the results form all of the mpi processes
        gathered_results = comm.gather(results, root=0)

        # export to csv make sure the name includes the n and the increment
        # for the csv file make sure to have the following columns: image_index, E1, <file_extension>_compression_ratio
        # name the csv file <compression_algo>_generator_calibration_data.csv 

        if rank == 0 and gathered_results is not None:

            final_results: List[Tuple[str, np.float16, np.float16, int, int, np.float16]] = [j for i in gathered_results for j in i]

            headers: List[str] = ["index", "entropy", "std", "size", "compressed_size", "compression_ratio"]

            filepath: Path = FileNamingTool.generate_filename("./assets/calibration", "npz-calibration-data", "csv", "npz")

            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(final_results)


        return 0 

    

if __name__ == "__main__":
    import time

    start = time.time()

    GenerateDeflateRelationship.npz_deflate()

    end = time.time()

    print('Execution time = %.6f seconds' % (end - start))


    
