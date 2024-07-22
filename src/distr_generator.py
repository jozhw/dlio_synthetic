from typing import List, Dict, Tuple
from operator import itemgetter

import random
import json

import pandas as pd
import numpy as np


class DistributionGenerator:

    @staticmethod
    def save_probabililites(probabilities, save_path="./data.json"):

        with open(save_path, "w") as file:

            json.dump(probabilities, file)

    @staticmethod
    def calculate_probabilities(occurrences: Dict[str, int], total) -> Dict[str, float]:
        # create a list of tuples with value and probability
        probabilities = {}
        for value, count in occurrences.items():
            probability = count / total
            probabilities[value] = probability

        return probabilities


    @staticmethod
    def get_rand_values(
        probabilities: Dict[str, float], size: int, 
    ) -> List[str]:

        rng = np.random.default_rng()

        keys: 	List[str] = [ key for key in probabilities.keys() ]
        arr = np.array(keys, dtype=object)
        values: List[float] = [value for value in probabilities.values()]

        print("Keys:", keys)
        print("Values:", values)
        print("Sum", sum(values))


        param = rng.choice(arr, size=size, p=values)
        #key2: List[str] =['1', '2', '3', '4']
        #prob = 0.5 - 0.1e-05
        #param = rng.choice(key2, size=10, p=[0.1e-05, prob, 0.2, 0.3])

        return param

    @staticmethod
    def get_params(
            df: pd.DataFrame, size: int, cr_type: str = "npz", to_round: int = 3
    ) -> Dict[str, List[str]]:

        usizes = df["uncompressed_size"].to_numpy()
        cr_name = "{}_compression_ratio".format(cr_type)
        crs = df[cr_name].to_numpy()

        # get length of the arrays
        l = len(usizes)

        occur_dims = {}

        occur_crs = {}

        for i in range(l):
            cr: float = round(crs[i], to_round)
            key_cr: str = f"{cr}"

            usize = usizes[i]
            # get the xside, which is one side of the square root of the the dimensions
            # to also use for greyscale, if the original size is not divisible by three,
            if usize % 3 != 0:
                size = usize

            # if divisible by three then has rgb channels
            else:
                size = usize / 3

            # get the side size of the to generate synthetic
            dim: int = int(np.sqrt(size))
            key_dim: str = f"{dim}"

            if key_cr in occur_crs:
                occur_crs[key_cr] += 1
            else:
                occur_crs[key_cr] = 1

            if key_dim in occur_dims:
                occur_dims[key_dim] += 1
            else:
                occur_dims[key_dim] = 1

        dim_probs = DistributionGenerator.calculate_probabilities(occur_dims, l)
        cr_probs = DistributionGenerator.calculate_probabilities(occur_crs, l)

        dim_param = DistributionGenerator.get_rand_values(dim_probs, size)
        cr_param = DistributionGenerator.get_rand_values(cr_probs, size)

        params: Dict[str, List[str]] = {
            "dimensions": dim_param,
            cr_name: cr_param
            
        }

        return params


def generate_x_values(x_occurrences, n):
    # calculate the total number of occurrences
    total_x = sum(x_occurrences.values())

    x_probabilities = DistributionGenerator.calculate_probabilities(
        x_occurrences, total_x
    )

    # generate the list of values given n
    x_values = DistributionGenerator.get_rand_values(x_probabilities, n)

    return x_values


if __name__ == "__main__":
    #x_occurrences = {"100": 1, "20": 10, "40": 1}
    #generated_values = generate_x_values(x_occurrences, 10)

    df_path = (
        "./results/20240626T192336==3=eagleimagenet--30000-processed-images-results.csv"
    )

    df = pd.read_csv(df_path)

    params = DistributionGenerator.get_params(df, 10)

    print(params)
