import json
import os
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from mpi4py import MPI
from PIL import Image

from calculations import Calculations as calc
from calculations import Entropy, EntropyCalculator
from compressions import Compressions as compr
from utils.filenamingtool import FileNamingTool
from utils.saving import Saving
from utils.validations import Validations


# decorator
def process_images(method):

    @wraps(method)
    def wrapper(self, *args, **kwargs):

        num_iter = 0
        comm = MPI.COMM_WORLD
        rank: int = comm.Get_rank()
        size: int = comm.Get_size()

        # only the 0th rank can load the data
        if rank == 0:
            data = self._load_json()
            paths = data["paths"]
        else:
            paths = None

        # broadcast paths to all nodes
        paths = comm.bcast(paths, root=0)

        # distribute the paths across processes
        paths_per_process = [[] for _ in range(size)]
        for i, path in enumerate(paths):
            paths_per_process[i % size].append(path)

        # to prevent duplications
        local_results = []

        # each process works on its assigned paths
        for path in paths_per_process[rank]:
            result = method(self, path)
            if result is not None:
                fname, data = result
                data["file_name"] = fname
                num_iter += 1

                local_results.append(data)

                # print(
                #    "Process {}, Completed iteration - {} for {}: entropy={}, compression_ratio={}".format(
                #        rank,
                #        num_iter,
                #        fname,
                #        data["entropy"],
                #        data["npz_compression_ratio"],
                #    )
                # )

        # gather results from all processes
        all_results = comm.gather(local_results, root=0)

        if rank == 0 and all_results is not None:

            # get number of images to be processed
            num_rows = len(paths)

            raw_filename = "{}-processed-images-results".format(num_rows)

            filepath = FileNamingTool.generate_filename(
                "./results/", raw_filename, "csv", self.source
            )

            # merge results from all processes
            flat_results = [result for sublist in all_results for result in sublist]

            # save to csv here

            Saving.save_to_csv(flat_results, filepath)

    return wrapper


class ImageProcessor:

    ACCEPTED_COMPRESSION_TYPES = ["npz", "jpg"]

    def __init__(self, source: str, json_file: str, compression_types: List[str]):
        # where the data comes from
        # for simplicity, the usecases for imagenet dataset on polaris is named polaris
        # the images collected locally will be named local
        self.source = source

        # json_file that has the image paths
        self.json = json_file
        # validate that a json file was inputed
        Validations.validate_json_extension(self.json)

        self.compression_types = compression_types
        # validate that the array of compression types are valid
        Validations.validate_compression_types(
            ImageProcessor.ACCEPTED_COMPRESSION_TYPES, self.compression_types
        )

        # determine the entropy classificiation

        self.window_size = 1
        self.shift_size = 1

        self.entropy_label = "E{}".format(self.window_size)
        self.shift_label = "S{}".format(self.shift_size)

    def _load_json(self):
        with open(self.json) as f:
            data: Dict = json.load(f)
        return data

    # uses self.compression_types
    @process_images
    def process_image(
        self,
        path,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:

        # get the filename itself without extensions
        filename, _ = os.path.splitext(os.path.basename(path))

        # set the values within the numpy array to uint8
        # may change this sometime in the future
        image: np.ndarray = np.array(Image.open(path)).astype(np.uint8)

        # numpy reads the image in the order of Height, Width, Depth
        dimensions: Tuple = image.shape

        height: int = dimensions[0]
        width: int = dimensions[1]

        # even though image processing can handle gray scale,
        # for the sake of consistency, gray scale will be filtered out
        if len(image.shape) == 2 or image.shape[2] != 3:
            return None

        # calculate the occurrences
        # this occurrence is still needed because of the mean intensity value
        occurrences: Dict[int, int] = calc.count_occurrences(image)
        mean: int = calc.calculate_mean_intensity_value(occurrences)

        ec = EntropyCalculator(window_size=1, shift_size=1, image=image)
        entropy: float = ec.calculate_entropy()

        total_pixels = dimensions[0] * dimensions[1]

        # the total size is in bytes, assuming an uint8 (range 0-255)
        uncompressed_size = total_pixels * dimensions[2]

        result: Dict[str, Any] = {
            self.entropy_label: entropy,
            "uncompressed_size": uncompressed_size,
            "uncompressed_height": height,
            "uncompressed_width": width,
            "mean_intensity_value": mean,
        }

        cresult: Dict[str, Any] = compr.compress_and_calculate(
            filename, image, dimensions, self.compression_types, self.source
        )

        # append the cresult to result
        for k, v in cresult.items():
            result[k] = v

        return filename, result

    # uses self.compression_types
    @process_images
    def process_rand_pixel_image(
        self,
        path,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        # get the filename itself without extensions
        filename, _ = os.path.splitext(os.path.basename(path))

        # add fand to filename
        filename = "randpixel-" + filename

        # set the values within the numpy array to uint8
        # may change this sometime in the future
        # create the image object
        image: np.ndarray = np.array(Image.open(path)).astype(np.uint8)

        # numpy reads the image in the order of Height, Width, Depth
        dimensions: Tuple = image.shape

        height: int = dimensions[0]
        width: int = dimensions[1]

        # even though image processing can handle gray scale,
        # for the sake of consistency, gray scale will be filtered out
        if len(image.shape) == 2 or image.shape[2] != 3:
            return None

        # randomize the object directly
        # since numpy only randomizes based on the axis

        arr = image.reshape((height * width, dimensions[2]))

        np.random.shuffle(arr)

        image = arr.reshape(dimensions)

        # calculate the occurrences
        occurrences: Dict[int, int] = calc.count_occurrences(image)
        mean: int = calc.calculate_mean_intensity_value(occurrences)
        ec = EntropyCalculator(
            window_size=self.window_size, shift_size=self.shift_size, image=image
        )
        entropy: float = ec.calculate_entropy()

        total_pixels = dimensions[0] * dimensions[1]

        # the total size is in bytes, assuming an uint8 (range 0-255)
        uncompressed_size = total_pixels * dimensions[2]

        result: Dict[str, Any] = {
            self.entropy_label: entropy,
            self.shift_label: self.shift_size,
            "uncompressed_size": uncompressed_size,
            "uncompressed_height": height,
            "uncompressed_width": width,
            "mean_intensity_value": mean,
        }

        cresult: Dict[str, Any] = compr.compress_and_calculate(
            filename, image, dimensions, self.compression_types, self.source
        )

        # append the cresult to result
        for k, v in cresult.items():
            result[k] = v

        return filename, result

    # uses self.compression_types
    @process_images
    def process_rand_image(
        self,
        path,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        # get the filename itself without extensions
        filename, _ = os.path.splitext(os.path.basename(path))

        # add fand to filename
        filename = "rand-" + filename

        # set the values within the numpy array to uint8
        # may change this sometime in the future
        # create the image object
        image: np.ndarray = np.array(Image.open(path)).astype(np.uint8)

        # numpy reads the image in the order of Height, Width, Depth
        dimensions: Tuple = image.shape

        height: int = dimensions[0]
        width: int = dimensions[1]

        # even though image processing can handle gray scale,
        # for the sake of consistency, gray scale will be filtered out
        if len(image.shape) == 2 or image.shape[2] != 3:
            return None

        # shuffle the image
        # first must flatten the array
        image = image.flatten()

        # randomize the object directly
        np.random.shuffle(image)

        # revert back to original dimensions
        image = np.reshape(image, dimensions)

        # calculate the occurrences
        occurrences: Dict[int, int] = calc.count_occurrences(image)
        mean: int = calc.calculate_mean_intensity_value(occurrences)

        ec = EntropyCalculator(
            window_size=self.window_size, shift_size=self.shift_size, image=image
        )

        entropy: float = ec.calculate_entropy()

        total_pixels = dimensions[0] * dimensions[1]

        # the total size is in bytes, assuming an uint8 (range 0-255)
        uncompressed_size = total_pixels * dimensions[2]

        result: Dict[str, Any] = {
            self.entropy_label: entropy,
            self.shift_label: self.shift_size,
            "uncompressed_size": uncompressed_size,
            "uncompressed_height": height,
            "uncompressed_width": width,
            "mean_intensity_value": mean,
        }

        cresult: Dict[str, Any] = compr.compress_and_calculate(
            filename, image, dimensions, self.compression_types, self.source
        )

        # append the cresult to result
        for k, v in cresult.items():
            result[k] = v

        return filename, result

    # uses self.compression_types
    @process_images
    def process_sorted_image(self, path) -> Optional[Tuple[str, Dict[str, Any]]]:

        # get the filename itself without extensions
        filename, _ = os.path.splitext(os.path.basename(path))

        # add fand to filename
        filename = "sorted-" + filename

        # set the values within the numpy array to uint8
        # may change this sometime in the future
        # create the image object
        image: np.ndarray = np.array(Image.open(path)).astype(np.uint8)

        # numpy reads the image in the order of Height, Width, Depth
        dimensions: Tuple = image.shape

        height: int = dimensions[0]
        width: int = dimensions[1]

        # even though image processing can handle gray scale,
        # for the sake of consistency, gray scale will be filtered out
        if len(image.shape) == 2 or image.shape[2] != 3:
            return None

        # sort the image using mergesort
        # first must flatten the array
        image = image.flatten()

        # sort with mergesort
        image = np.sort(image, kind="mergesort")

        # revert back to original dimensions
        image = image.reshape(dimensions)

        # calculate the occurrences
        occurrences: Dict[int, int] = calc.count_occurrences(image)
        mean: int = calc.calculate_mean_intensity_value(occurrences)
        ec = EntropyCalculator(
            window_size=self.window_size, shift_size=self.shift_size, image=image
        )

        entropy: float = ec.calculate_entropy()

        total_pixels = dimensions[0] * dimensions[1]

        # the total size is in bytes, assuming an uint8 (range 0-255)
        uncompressed_size = total_pixels * dimensions[2]

        result: Dict[str, Any] = {
            self.entropy_label: entropy,
            self.shift_label: self.shift_size,
            "uncompressed_size": uncompressed_size,
            "uncompressed_height": height,
            "uncompressed_width": width,
            "mean_intensity_value": mean,
        }

        cresult: Dict[str, Any] = compr.compress_and_calculate(
            filename, image, dimensions, self.compression_types, self.source
        )

        # append the cresult to result
        for k, v in cresult.items():
            result[k] = v

        return filename, result

    # uses self.compression_types
    @process_images
    def process_resized_image(
        self,
        path,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:

        # get the filename itself without extensions
        filename, _ = os.path.splitext(os.path.basename(path))

        # set the values within the numpy array to uint8
        # may change this sometime in the future
        oimage = cv2.imread(path)

        # cv2 uses BGR, convert to RGB
        oimage = cv2.cvtColor(oimage, cv2.COLOR_BGR2RGB)

        dim = (224, 224)

        resized = cv2.resize(oimage, dim, interpolation=cv2.INTER_AREA)

        image: np.ndarray = np.array(resized).astype(np.uint8)

        # numpy reads the image in the order of Height, Width, Depth
        dimensions: Tuple = image.shape

        height: int = dimensions[0]
        width: int = dimensions[1]

        # even though image processing can handle gray scale,
        # for the sake of consistency, gray scale will be filtered out
        if len(image.shape) == 2 or image.shape[2] != 3:
            return None

        # calculate the occurrences
        # this occurrence is still needed because of the mean intensity value
        occurrences: Dict[int, int] = calc.count_occurrences(image)
        mean: int = calc.calculate_mean_intensity_value(occurrences)

        ec = EntropyCalculator(
            window_size=self.window_size, shift_size=self.shift_size, image=image
        )
        entropy: float = ec.calculate_entropy()

        total_pixels = dimensions[0] * dimensions[1]

        # the total size is in bytes, assuming an uint8 (range 0-255)
        uncompressed_size = total_pixels * dimensions[2]

        result: Dict[str, Any] = {
            self.entropy_label: entropy,
            self.shift_label: self.shift_size,
            "uncompressed_size": uncompressed_size,
            "uncompressed_height": height,
            "uncompressed_width": width,
            "mean_intensity_value": mean,
        }

        cresult: Dict[str, Any] = compr.compress_and_calculate(
            filename, image, dimensions, self.compression_types, self.source
        )

        # append the cresult to result
        for k, v in cresult.items():
            result[k] = v

        return filename, result

    # uses self.compression_types
    @process_images
    def process_resized_imageEn(
        self,
        path,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:

        # get the filename itself without extensions
        filename, _ = os.path.splitext(os.path.basename(path))

        # set the values within the numpy array to uint8
        # may change this sometime in the future
        oimage = cv2.imread(path)

        # cv2 uses BGR, convert to RGB
        oimage = cv2.cvtColor(oimage, cv2.COLOR_BGR2RGB)

        dim = (224, 224)

        resized = cv2.resize(oimage, dim, interpolation=cv2.INTER_AREA)

        image: np.ndarray = np.array(resized).astype(np.uint8)

        # numpy reads the image in the order of Height, Width, Depth
        dimensions: Tuple = image.shape

        height: int = dimensions[0]
        width: int = dimensions[1]

        # even though image processing can handle gray scale,
        # for the sake of consistency, gray scale will be filtered out
        if len(image.shape) == 2 or image.shape[2] != 3:
            return None

        flattened_image = image.flatten()
        # calculate the occurrences

        E = Entropy.E2(flattened_image)

        total_pixels = dimensions[0] * dimensions[1]

        # the total size is in bytes, assuming an uint8 (range 0-255)
        uncompressed_size = total_pixels * dimensions[2]

        result: Dict[str, Any] = {
            self.entropy_label: E,
            "uncompressed_size": uncompressed_size,
            "uncompressed_height": height,
            "uncompressed_width": width,
        }

        cresult: Dict[str, Any] = compr.compress_and_calculate(
            filename, image, dimensions, self.compression_types, self.source
        )

        # append the cresult to result
        for k, v in cresult.items():
            result[k] = v

        return filename, result

    # uses self.compression_types
    @process_images
    def process_imageEn(
        self,
        path,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:

        # get the filename itself without extensions
        filename, _ = os.path.splitext(os.path.basename(path))

        # set the values within the numpy array to uint8
        # may change this sometime in the future
        oimage = cv2.imread(path)

        # cv2 uses BGR, convert to RGB
        oimage = cv2.cvtColor(oimage, cv2.COLOR_BGR2RGB)

        image: np.ndarray = np.array(Image.open(path)).astype(np.uint8)

        # numpy reads the image in the order of Height, Width, Depth
        dimensions: Tuple = image.shape

        height: int = dimensions[0]
        width: int = dimensions[1]

        # even though image processing can handle gray scale,
        # for the sake of consistency, gray scale will be filtered out
        if len(image.shape) == 2 or image.shape[2] != 3:
            return None

        red_channel = image[:, :, 0].flatten()

        E = Entropy.E3(red_channel)

        total_pixels = dimensions[0] * dimensions[1]

        # the total size is in bytes, assuming an uint8 (range 0-255)
        uncompressed_size = total_pixels * dimensions[2]

        result: Dict[str, Any] = {
            self.entropy_label: E,
            "uncompressed_size": uncompressed_size,
            "uncompressed_height": height,
            "uncompressed_width": width,
        }

        cresult: Dict[str, Any] = compr.compress_and_calculate(
            filename, image, dimensions, self.compression_types, self.source
        )

        # append the cresult to result
        for k, v in cresult.items():
            result[k] = v

        return filename, result


if __name__ == "__main__":
    # json_path = "./results/image_paths/test_paths.json"
    json_path = (
        "results/image_paths/20240625T202523==3=eagleimagenet--imagenet-rand-30000.json"
    )

    compression_types = ["npz"]

    imgp = ImageProcessor("eagleimagenet", json_path, compression_types)

    imgp.process_image()
