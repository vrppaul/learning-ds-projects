import numpy as np


class Perceptron:
    def __init__(self, X: list, y: list, init_type: str = 'zeros',
                 threshold: float = 0.5, learning_rate: float = 0.1, max_epochs: int = 10):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.X = X
        self.y = y
        self.max_epochs = max_epochs
        self.initialize(init_type)

    def initialize(self, init_type: str) -> None:
        if init_type == 'random':
            self.weights = np.random.rand(len(self.X[0])) * 0.05
        elif init_type == 'zeros':
            self.weights = np.zeros(len(self.X[0]))
        print('\nSuccessful initialization!')
        print(f'Initialized weights are {self.weights}')

    def train(self) -> None:
        epoch = 0
        while True:
            error_count = 0
            epoch += 1
            print(f'\n---- EPOCH NUMBER {epoch} ----\n')
            rows = list(zip(self.X, self.y))
            np.random.shuffle(rows)
            for X, y in rows:
                error_count += self.train_observation(X, y)
            if error_count == 0:
                print('\nTraining successful!')
                break
            if epoch >= self.max_epochs:
                print('\nReached maximum epochs. No perfect prediction')
                break

    def train_observation(self, X: tuple, y: int) -> int:
        error_count = 0
        result = np.dot(X, self.weights) > self.threshold
        error = y - result
        if error != 0:
            error_count += 1
            for index, value in enumerate(X):
                self.weights[index] += self.learning_rate * error * value
            print('\nWeights have been changed')
            print(f'New weights are {self.weights}')

        return error_count

    def predict(self, X: tuple) -> int:
        return int(np.dot(X, self.weights) > self.threshold)

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return self.__str__()
