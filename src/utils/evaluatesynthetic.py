from typing import Dict, List

import pandas as pd
from filenamingtool import FileNamingTool
from validations import Validations


class EvaluateSynthetic:

    ACCEPTED_COMPRESSED_FILE_TYPES = ["npz", "jpg"]

    def __init__(
        self,
        original_data: str,
        synthetic_data: str,
        sample_size: int,
        num_iterations: int,
        compressed_file_types: List[str],
        source,
    ):

        self.compressed_file_types = compressed_file_types

        # validate if compressed file types are accepted
        Validations.validate_compressed_file_types(
            EvaluateSynthetic.ACCEPTED_COMPRESSED_FILE_TYPES, compressed_file_types
        )

        self.o_path = original_data
        self.odf = pd.read_csv(self.o_path)

        self.s_path = synthetic_data
        self.sdf = pd.read_csv(self.s_path)

        self.sample_size = sample_size
        self.iter = num_iterations

        # where did the data originate? (example: polaris, local)
        self.source = source

        self.results: List[Dict] = []

    def _calculate_uncompressed_diff(self) -> Dict:

        results = {}

        uosize_column = self.odf["uncompressed_size"]
        sample_uosize = uosize_column.sample(self.sample_size)
        total_sample_uosize: int = sum(sample_uosize)

        total_ussize: int = sum(self.sdf["uncompressed_size"])

        difference: int = abs(total_sample_uosize - total_ussize)

        ratio: float = total_ussize / total_sample_uosize

        results = {
            "total_original_uncompressed_size": total_sample_uosize,
            "total_synthetic_uncompressed_size": total_ussize,
            "total_uncompressed_difference": difference,
            "total_uncompressed_ratio": ratio,
        }

        return results

    def _calculate_compressed_diff(self) -> Dict[str, Dict]:
        """
        Given that the column naming convention of compressed data is <type>_compressed_image_size,
        a iteration through the type without having to use custom methods for each type can be
        used.

        """

        results = {}

        for t in self.compressed_file_types:

            # get the right data column
            query = "{}_compressed_image_size".format(t)
            # original compressed type size
            ocsize: int = sum(self.odf[query])
            # synthetic compressed type size
            scsize: int = sum(self.sdf[query])

            # absolute difference
            difference: int = abs(ocsize - scsize)

            # ratio
            ratio: float = scsize / ocsize

            key_total_original = "total_original_{}_compressed_size".format(t)
            key_total_synthetic = "total_synthetic_{}_compressed_size".format(t)
            key_difference = "total_{}_compressed_difference".format(t)
            key_ratio = "total_{}_compressed_ratio".format(t)

            results[key_total_original] = ocsize
            results[key_total_synthetic] = scsize
            results[key_difference] = difference
            results[key_ratio] = ratio

        return results

    def calculate_diff(self) -> None:

        uresults = self._calculate_uncompressed_diff()
        cresults = self._calculate_compressed_diff()

        # merge cresults with uresults
        uresults.update(cresults)

        # append the results to the array object
        self.results.append(uresults)

    def save_results(self) -> None:

        df = pd.DataFrame(self.results)
        filename = FileNamingTool.generate_filename(
            "./results", "evaluate-synthetic", "csv", self.source
        )
        df.to_csv(filename, index=False)

        ratio_df = df.filter(regex="ratio")
        # calculate descriptive statistics that includes mean, quartiles, etc.
        summary = ratio_df.describe()

        rfilename = FileNamingTool.generate_filename(
            "./results", "evaluate-synthetic-summary-statistics", "txt", self.source
        )

        with open(rfilename, "w") as f:
            f.write(summary.to_string())

        print(summary)
