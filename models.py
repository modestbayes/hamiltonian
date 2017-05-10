import numpy as np

class Logistic:
    """
    Logistic regression model.
    """

    def __init__(self, data_X, data_y, alpha):
        self.X = data_X
        self.y = data_y
        self.alpha = alpha

    def __loglikelihood(self, beta):
        """
        Log logistic regression likelihood of beta.
        """
        return(-np.sum(np.log(np.exp(np.dot(self.X, beta)) + 1.0)) \
            + np.dot(np.transpose(self.y), np.dot(self.X, beta)))

    def __prior(self, beta):
        """
        Log independent Gaussian prior density of beta.
        """
        return(-(0.5 / self.alpha) * np.sum(np.square(beta)))

    def get_dim(self):
        return(self.X.shape[1])

    def energy(self, beta):
        return(-(self.__loglikelihood(beta) + self.__prior(beta)))

    def gradient(self, beta):
        return(np.dot(np.transpose(self.X), self.y - np.exp(np.dot(self.X, beta)) \
            / (1.0 + np.exp(np.dot(self.X, beta)))) - beta / self.alpha)

class Gaussian:
    """
    Gaussian distribution.
    """

    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def __logL_gaussian(self, x):
        return(-0.5 * np.sum((x - self.mu) * (1.0 / self.var) * (x - self.mu)) - 0.5 * np.log(np.prod(self.var)))

    def sample(self, n):
        dim = self.mu.shape[0]
        samples = np.zeros((n, dim))
        for i in range(dim):
            samples[:, i] = np.random.normal(self.mu[i], np.sqrt(self.var[i]), n)
        return(samples)

    def get_dim(self):
        return(self.mu.shape[0])

    def energy(self, x):
        return(-self.__logL_gaussian(x))

    def gradient(self, x):
        return(-(x - self.mu) / self.var)
    
class Riemannian:
    """
    Gaussian distribution with Riemannian metric (hessian).
    """

    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def __logL_gaussian(self, x):
        return(-0.5 * np.sum((x - self.mu) * (1.0 / self.var) * (x - self.mu)) - 0.5 * np.log(np.prod(self.var)))

    def sample(self, n):
        dim = self.mu.shape[0]
        samples = np.zeros((n, dim))
        for i in range(dim):
            samples[:, i] = np.random.normal(self.mu[i], np.sqrt(self.var[i]), n)
        return(samples)

    def get_dim(self):
        return(self.mu.shape[0])

    def energy(self, x):
        return(-self.__logL_gaussian(x))

    def gradient(self, x):
        return(-(x - self.mu))