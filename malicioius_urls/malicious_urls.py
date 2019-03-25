import tarfile

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
import numpy as np


uri = '/home/bss_pracant/learning/malicioius_urls/data/url_svmlight.tar.gz'
tar = tarfile.open(uri, 'r:gz')
max_observations = 0
max_variables = 0
i = 0
split = 5

for tarinfo in tar:
    print(f'Extracting {tarinfo.name}, file size is {tarinfo.size}')
    if tarinfo.isfile():
        file = tar.extractfile(tarinfo.name)
        X, y = load_svmlight_file(file)
        max_variables = np.maximum(max_variables, X.shape[0])
        max_observations = np.maximum(max_observations, X.shape[1])

    if i > split:
        break
    i += 1

print(f'Max X = {max_observations}, max y dimension = {max_variables}')