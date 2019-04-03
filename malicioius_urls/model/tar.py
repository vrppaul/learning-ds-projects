import tarfile
from itertools import islice

from sklearn.datasets import load_svmlight_file
import numpy as np


class Tar:
    URI: str
    TAR: tarfile.TarFile
    CLASSES: [int]

    def __init__(self, uri: str, classes: [int]):
        self.URI = uri
        self.TAR = tarfile.open(self.URI, 'r:gz')
        self.CLASSES = classes

    def get_tar_files(self, max_files: int):
        return islice(filter(lambda item: item.isfile(), self.TAR), max_files)

    @staticmethod
    def get_maximum_variables(tar, tar_files) -> (int, int):
        print('\n---- COUNTING VARIABLES ----')
        max_variables = 0
        for tarinfo in tar_files:
            file = tar.extractfile(tarinfo.name)
            X, _ = load_svmlight_file(file)
            max_variables = np.maximum(max_variables, X.shape[1])

        return max_variables
