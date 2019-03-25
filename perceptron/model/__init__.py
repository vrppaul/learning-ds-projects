import numpy as np


class Perceptron:
    """
    Perceptron

    A perceptron is one of the least complex machine learning algorithms
    used for binary classification (0 or 1): for instance, will the customer buy or not.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    init_type: str
        Weights initialization method.
        'zeros' means that, the weight matrix will be filled with zeros
        'random' means that, the weight matrix will be filled with small numbers between 0 and 0.05
        Default is 'zeros'

    threshold: float
        Threshold to determine, whether the result belongs to 0 or 1

    learning_rate: float
        How fast and precise does the perceptron learn

    max_epochs: int
        How many unsuccessful can be made before learning break
    """
    def __init__(self, X: list, y: list, init_type: str = 'zeros',
                 threshold: float = 0.5, learning_rate: float = 0.1, max_epochs: int = 10):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.X = X
        self.y = y
        self.max_epochs = max_epochs
        self.initialize(init_type)

    def initialize(self, init_type: str) -> None:
        """Initialize the Perceptron object"""
        if init_type == 'random':
            self.weights = np.random.rand(len(self.X[0])) * 0.05
        elif init_type == 'zeros':
            self.weights = np.zeros(len(self.X[0]))
        print('\nSuccessful initialization!')
        print(f'Initialized weights are {self.weights}')

    def train(self) -> None:
        """Train the Perceptron object on given training data"""
        epoch = 0
        while True:
            error_count = 0
            epoch += 1
            print(f'\n---- EPOCH NUMBER {epoch} ----\n')

            # Creating convenient data format for iterating
            rows = list(zip(self.X, self.y))
            # Randomizing data order to prevent overfitting
            np.random.shuffle(rows)
            # Iterating and learning (tweaking weights) over all samples and counting errors
            for X, y in rows:
                error_count += self.train_observation(X, y)

            # If no errors were made during learning, the model is perfectly fit
            # Therefore there is no point to tweak weights anymore
            if error_count == 0:
                print('\nTraining successful!')
                break
            # If maximum epochs number is reached and no perfect model was found
            if epoch >= self.max_epochs:
                print('\nReached maximum epochs. No perfect prediction')
                break

    def train_observation(self, X: tuple, y: int) -> int:
        """Train model (tweak weights) by observing single sample"""
        error_count = 0

        # Determine, whether a predicted train label belongs to 0 or 1
        result = np.dot(X, self.weights) > self.threshold
        # Comparing to a real label
        error = y - result
        if error != 0:
            error_count += 1
            # Tweaking all the weights one by one
            for index, value in enumerate(X):
                self.weights[index] += self.learning_rate * error * value
            print('\nWeights have been changed')
            print(f'New weights are {self.weights}')

        return error_count

    def predict(self, X: tuple) -> int:
        """Predicting a sample label of an unseen sample"""
        return int(np.dot(X, self.weights) > self.threshold)

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return self.__str__()
