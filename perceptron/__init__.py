from perceptron.model import Perceptron


if __name__ == '__main__':
    X = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (1, 0, 1), (1, 0, 1)]
    y = [1, 1, 0, 0, 1, 1]

    perceptron = Perceptron(X, y, 'random')
    perceptron.train()

    print(perceptron.predict((1, 1, 1)))
    print(perceptron.predict((1, 0, 1)))
