import numpy as np

class ShallowNet:
    """
    Shallow random projection neural network.
    
    The neural network has two sets of weights and biases: W1, b1, W2.
    W1 and b1 are randomized while W2 are trained with penalized regression.
    
    Attributes
        dim: dimension of input vector
        size: size of hidden layer
        W1, b1, W2: neural network parameters
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
        
        Args
            X: 2d numpy array (n x p) of input matrix
        
        Returns
            2d numpy array of hidden layer output
        """
        return(np.log(1.0 + np.exp(np.dot(X, np.transpose(self.W1)) + self.b1)))

    def train(self, X, y, penalty):
        """
        Train output weights with penalty.
        
        Args
            X: 2d numpy array (n x p) of training points
            y: 1d numpy array (n) of target vector
            penalty: weight of L2 penalty
        """
        H = self.__hidden(X)
        hat = np.dot(np.transpose(H), H) + np.identity(self.size) * penalty
        weights = np.dot(np.linalg.inv(hat), np.dot(np.transpose(H), y))
        self.W2 = weights

    def gradient(self, x):
        """
        Calculate gradient of neural network.
        
        Args
            x: 1d numpy array of input vector
        
        Returns
            1d numpy array of gradient vector
        """
        hidden_grad = 1.0 / (1.0 + np.exp(-(np.dot(self.W1, x) + self.b1)))
        return((np.dot(np.transpose(self.W2) * hidden_grad, self.W1)).reshape(self.dim))

class NeuralGrad:
    """
    Neural network gradient approximator in Keras.
    
    Attributes:
        network: pre-trained Keras model
        preprocessor: sklearn preprocessor 
    """

    def __init__(self, network, preprocessor):
        self.network = network
        self.preprocessor = preprocessor

    def gradient(self, x):
        """
        Predict gradient with neural network.
        
        Args
            x: 1d numpy array of input vector
        
        Returns
            1d numpy array of gradient vector
        """
        if x.shape[0] != 1:
            x = x.reshape(1, x.shape[0])
        x_scaled = self.preprocessor.transform(x)
        return(self.network.predict(x_scaled)[0])
    
class GaussianProcess:
    """
    Gaussian process surrogate.
    
    Attributes
        kernel: sklearn GP kernel 
        training: 2d numpy array (n x p) of training points
        energy: 1d numpy array (n) of target vector
        energy_normalized: 1d numpy array (n) of normalized energy
        length: length scale parameter of squared exponential kernel
        K_inverse: inverted observed kernel matrix
    """

    def __init__(self, kernel, training, energy, length):
        self.kernel = kernel # sklearn GP kernel
        self.training = training # data X
        self.energy = energy # data Y
        self.energy_normalized = (energy - np.mean(energy)) / np.std(energy)
        self.length = length # GP parameter for calculating derivative
        self.K_inverse = np.linalg.inv(kernel(training)) # inverse kernel matrix

    def gradient(self, x):
        """
        Calculate gradient of GP.
        
        Args
            x: 1d numpy array of input vector
        
        Returns
            1d numpy array of gradient vector
        """
        n = self.K_inverse.shape[0]
        k = self.training.shape[1]
        # calculate gradient first component with chain rule
        C = self.kernel(x, self.training) 
        dk = np.zeros((k, n))
        for i in range(k):
            dk[i, :] = - C * (self.training[:, i] - x[i]) / self.length ** 2
        return(np.dot(np.dot(dk, self.K_inverse), self.energy_normalized) * np.mean(self.energy))

class Stochastic:
    """
    Stochastic gradient.
    
    Attributes
        X, y: data
        alpha: prior parameter
        n: total number of observations
        k: mini-batch size
    """

    def __init__(self, model, size):
        self.X = model.X
        self.y = model.y
        self.alpha = model.alpha
        self.n = model.y.shape[0]
        self.k = size

    def gradient(self, beta):
        """
        Calculate stochastic gradient.
        
        Args
            x: 1d numpy array of input vector
        
        Returns
            1d numpy array of gradient vector
        """
        select = np.random.randint(low=0, high=self.n, size=self.k)
        subsetX = self.X[select, :]
        subsety = self.y[select]
        return(np.dot(np.transpose(subsetX), subsety - np.exp(np.dot(subsetX, beta)) \
            / (1.0 + np.exp(np.dot(subsetX, beta)))) - beta / self.alpha)
