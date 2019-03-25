from perceptron.model import Perceptron


if __name__ == '__main__':
    # Manually creating inputs and outputs
    X = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (1, 0, 1), (1, 0, 1)]
    y = [1, 1, 0, 0, 1, 1]

    # Initializing and training new Perceptron object
    perceptron = Perceptron(X, y, 'random')
    perceptron.train()

    # Predicting outputs
    print(perceptron.predict((1, 1, 1)))
    print(perceptron.predict((1, 0, 1)))
