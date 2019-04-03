from malicioius_urls.model.tar import Tar

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier


MAX_FILES = 20
TAR = Tar(uri='/home/bss_pracant/learning/malicioius_urls/data/url_svmlight.tar.gz',
          classes=[-1, 1])

max_variables = Tar.get_maximum_variables(TAR.TAR, TAR.get_tar_files(MAX_FILES))
print(f'\nMAX VARIABLES = {max_variables}')

sgd = SGDClassifier(loss='log')

for index, tarinfo in enumerate(TAR.get_tar_files(MAX_FILES)):
    file = TAR.TAR.extractfile(tarinfo.name)
    X, y = load_svmlight_file(file, n_features=max_variables)
    print(f'\nTraining on file #{index}')
    sgd.partial_fit(X, y, classes=TAR.CLASSES)
    
    if index == MAX_FILES - 1:
        print('\n---- TRAINING RESULTS ----')
        print(classification_report(sgd.predict(X), y))
