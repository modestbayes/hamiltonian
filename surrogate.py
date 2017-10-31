import numpy as np

class Shallow:
    """
    Shallow random projection neural network.
    """

    def __init__(self, dim, size):
        self.dim = dim
        self.size = size
        self.W1 = (np.random.normal(0, 1, size * dim)).reshape((size, dim))
        self.b1 = np.random.normal(0, 1, size)
        self.W2 = np.zeros((size, 1))

    def __hidden(self, X):
        """
        Forward pass input to hidden layer.
        """
        return(np.log(1.0 + np.exp(np.dot(X, np.transpose(self.W1)) + self.b1)))

    def train(self, X, y, penalty):
        """
        Train output weights with penalty.
        """
        H = self.__hidden(X)
        hat = np.dot(np.transpose(H), H) + np.identity(self.size) * penalty
        weights = np.dot(np.linalg.inv(hat), np.dot(np.transpose(H), y))
        self.W2 = weights

    def gradient(self, x):
        """
        Calculate neural network gradient.
        """
        hidden_grad = 1.0 / (1.0 + np.exp(-(np.dot(self.W1, x) + self.b1)))
        return((np.dot(np.transpose(self.W2) * hidden_grad, self.W1)).reshape(self.dim))

class KerasGrad:
    """
    Neural network gradient approximator with Keras.
    """

    def __init__(self, network, preprocessor):
        self.network = network
        self.preprocessor = preprocessor

    def gradient(self, x):
        if x.shape[0] != 1:
            x = x.reshape(1, x.shape[0])
        x_scaled = self.preprocessor.transform(x)
        return(self.network.predict(x_scaled)[0])

class NeuralGrad:
    """
    Neural network gradient approximator.
    """

    def __init__(self, W1, b1, W2, b2, preprocessor):
        self.weights1 = W1
        self.biases1 = b1
        self.weights2 = W2
        self.biases2 = b2
        self.mean = preprocessor.mean_
        self.std = np.sqrt(preprocessor.var_)

    def gradient(self, x):
        x_scaled = (x - self.mean) / self.std
        hidden = np.dot(self.weights1, x_scaled) + self.biases1
        hidden = hidden * (hidden > 0)
        output = np.dot(self.weights2, hidden) + self.biases2
        return(output)

class StoGrad:
    """
    Stochastic gradient.
    """

    def __init__(self, model, size):
        self.X = model.X
        self.y = model.y
        self.alpha = model.alpha
        self.n = model.y.shape[0]
        self.k = size

    def gradient(self, beta):
        select = np.random.randint(low=0, high=self.n, size=self.k)
        subsetX = self.X[select, :]
        subsety = self.y[select]
        return(np.dot(np.transpose(subsetX), subsety - np.exp(np.dot(subsetX, beta)) \
            / (1.0 + np.exp(np.dot(subsetX, beta)))) - beta / self.alpha)
    
class GaussianProcess:
    """
    Gaussian process surrogate.
    """

    def __init__(self, kernel, training, energy, length):
        self.kernel = kernel # sklearn GP kernel
        self.training = training # data X
        self.energy = energy # data Y
        self.energy_normalized = (energy - np.mean(energy)) / np.std(energy)
        self.length = length # GP parameter for calculating derivative
        self.K_inverse = np.linalg.inv(kernel(training)) # inverse kernel matrix

    def gradient(self, x):
        n = self.K_inverse.shape[0]
        k = self.training.shape[1]
        # calculate gradient first component with chain rule
        C = self.kernel(x, self.training) 
        dk = np.zeros((k, n))
        for i in range(k):
            dk[i, :] = - C * (self.training[:, i] - x[i]) / self.length ** 2
        return(np.dot(np.dot(dk, self.K_inverse), self.energy_normalized) * np.mean(self.energy))
