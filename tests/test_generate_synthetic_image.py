import os
import sys
import unittest

import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))

# attach the src too path to access the tools module
sys.path.append(src_root)

from generate_synthetic_image import GenerateSyntheticImage as gen


class TestGenerateSyntheticImage(unittest.TestCase):

    def test_sort_synthetic_image(self):

        test_arr = np.array([2, 1, 3, 5, 4])

        sorted_arr = np.array([1, 2, 3, 4, 5])

        result = gen._sort_synthetic_image(test_arr) == sorted_arr

        self.assertTrue(result.all())


if __name__ == "__main__":
    unittest.main()
