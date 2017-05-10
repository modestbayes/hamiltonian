import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_data(n, beta, logistic=False, sigma=1.0):
    """Generate regression data.
    
    # Arguments
        n: number of observations
        beta: regression coefficient vector
        logistic: logsitic or linear regression
        sigma: standard deviation of Gaussian noise (for linear regression)
        
    # Returns
        A list of design matrix and response vector
    """
    d = beta.shape[0]
    X = np.random.normal(0, 1, n * d).reshape(n, d)
    eta = np.dot(X, beta)
    if logistic:
        mu = 1.0 / (1.0 + np.exp(-eta))
        Y = np.random.binomial(1, mu) * 1.0
    else:
        mu = eta
        Y = eta + np.random.normal(0, sigma, n).reshape(n, 1)
    return(X, Y)

def diagnostic_plots(chain):
    """
    Plot the trace and autocorrelation of a Markov chain (posterior draws of one variable).
    """
    fig = plt.figure(figsize=(8, 2), dpi=200)
    plt.plot(chain)
    plt.title('Trace')

    fig = plt.figure(figsize=(8, 2), dpi=200)
    plt.ylim(0, 1)
    maxlag = 50
    n = chain.shape[0]
    acf = np.zeros(maxlag + 1)
    for t in range(maxlag + 1):
        acf[t] = np.corrcoef(chain[t:], chain[:(n - t)])[0, 1]
    plt.plot(acf, linestyle='dashed', marker='.', color='red')
    plt.title('Autocorrelation')

def effective_n(chain):
    """
    Calculate the effective sample size of a Markov chain.
    """
    maxlag = 200
    n = chain.shape[0]
    rho = np.zeros(maxlag)
    for i in range(maxlag):
        t = i + 1
        acf = np.corrcoef(chain[t:], chain[:(n - t)])[0, 1]
        if acf < 0:
            break
        rho[i] = acf
    return(n / (1 + np.sum(rho)))

def effective_n_all(chains):
    """
    Call effective_n on multiple chains.
    """
    k = chains.shape[1]
    size = np.zeros(k)
    for i in range(k):
        size[i] = effective_n(chains[:, i])
    return(size)

def scatter_matrix(chain1, chain2):
    """
    Plot the marginal and joint posterior distributions of two variables.
    """
    fig = plt.figure(figsize=(8, 3), dpi=200)
    plt.subplot(121)
    plt.hist(chain1, bins=50, alpha=0.5, color='green')

    plt.subplot(122)
    plt.plot(chain1, chain2, 'o', alpha=0.5)

    fig = plt.figure(figsize=(8, 3), dpi=200)
    plt.subplot(121)
    plt.plot(chain2, chain1, 'o', alpha=0.5)

    plt.subplot(122)
    plt.hist(chain2, bins=50, alpha=0.5, color='red')