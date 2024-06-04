import copy
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from tools.filenamingtool import FileNamingTool
from tools.removing import Removing


class SpatialInfluence:

    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        self.results = []

    @staticmethod
    def _verify_lossless_spacial_influence(image_path):

        # get the file name with the extension
        raw_fname = os.path.basename(image_path)

        # get the file name itself (without the extension)
        fname = raw_fname.split(".")[0]

        # load image into numpy array
        image: np.ndarray = np.array(Image.open(image_path)).astype(np.uint8)

        # save the shape to be used to reshape the shuffled image
        shape = image.shape

        # original image compressed to npz
        np.savez_compressed("{}.npz".format(fname), image)
        # get size of the original npz compressed format
        unshuffled_size = os.path.getsize("{}.npz".format(fname))

        # shuffled image
        shuffled_image = copy.deepcopy(image)

        np.random.shuffle(shuffled_image.reshape(-1))

        # reshape the shuffled image to original image dimensions
        shuffled_image = np.array(shuffled_image, dtype=np.uint8).reshape(shape)

        # save the shuffled image that is compressed into a npz file
        np.savez_compressed("shuffled_{}.npz".format(fname), shuffled_image)

        # get the size of the shuffled compressed iamge
        shuffled_size = os.path.getsize("shuffled_{}.npz".format(fname))

        # shuffled2_image
        shuffled2_image = copy.deepcopy(shuffled_image.reshape(-1))

        # the second shuffling
        np.random.shuffle(shuffled2_image)

        # reshape the shuffled2_image to original image dimensions
        shuffled2_image = np.array(shuffled2_image, dtype=np.uint8).reshape(shape)

        # save the shuffled2_image that is compressed into a npz file
        np.savez_compressed("shuffled2_{}.npz".format(fname), shuffled2_image)

        # get the size of the shuffled2 compressed image
        shuffled2_size = os.path.getsize("shuffled2_{}.npz".format(fname))

        Removing.remove_compressed_imgs("{}.npz".format(fname))
        Removing.remove_compressed_imgs("shuffled_{}.npz".format(fname))
        Removing.remove_compressed_imgs("shuffled2_{}.npz".format(fname))

        # ratio of the npz compressed shuffled and unshuffled sizes
        ratio = shuffled_size / unshuffled_size

        # ratio of the npz compressed shuffled2 and unshuffled sizes
        ratio_s2 = shuffled2_size / unshuffled_size

        # ratio of the npz compressed shuffled 2 and the shuffled sizes
        sratio = shuffled2_size / shuffled_size

        return fname, ratio, ratio_s2, sratio

    def run(self):

        results: List[Dict[str, float]] = []

        for image_path in self.image_paths:
            fname, npz_ratio, npz_ratio_s2, npz_sratio = (
                SpatialInfluence._verify_lossless_spacial_influence(image_path)
            )

            # results object for each iteration that will be appended to the overall
            # results object
            obj = {
                "filename": fname,
                "npz_ratio": npz_ratio,
                "npz_ratio_s2": npz_ratio_s2,
                "npz_sratio": npz_sratio,
            }
            results.append(obj)

        self.results = results

    def save_to_csv(self):

        length = len(self.results)

        df = pd.DataFrame(self.results)

        fname_csv = FileNamingTool.generate_filename(
            "./results",
            "{}-images-spatial-influence-results".format(length),
            "csv",
            "local",
        )

        df.to_csv(fname_csv, index=False)

        ratio_df = df.filter(regex="ratio")
        # calculate descriptive statistics that includes mean, quartiles, etc.
        summary = ratio_df.describe()

        rfilename = FileNamingTool.generate_filename(
            "./results",
            "{}-images-spatial-influence-summary-statistics".format(length),
            "txt",
            "local",
        )

        with open(rfilename, "w") as f:
            f.write(summary.to_string())

        print(summary)


if __name__ == "__main__":
    import glob
    import sys

    paths = glob.glob(
        "/Users/johnz.wu/Projects/Code/dlio_synthetic/assets/test_images/*.jpg"
    )

    si = SpatialInfluence(paths)

    si.run()

    si.save_to_csv()
