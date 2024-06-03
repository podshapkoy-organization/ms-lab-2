import numpy as np
from scipy.stats import norm


def normal_sample(n, mu, sigma):
    return np.random.normal(mu, sigma, n)


def aprior(theta, mu, sigma):
    return norm.pdf(theta, loc=mu, scale=sigma)


def bayesian_estimate(samples, mu0, sigma0):
    n = len(samples)
    mean_sample = np.mean(samples)
    sample_var = np.var(samples)
    sigma = np.sqrt(sample_var)

    def posterior(theta):
        return aprior(theta, mu0, sigma0) * norm.pdf(mean_sample, loc=theta, scale=sigma / np.sqrt(n))

    normalization_constant = np.trapz([posterior(theta) for theta in np.linspace(-10, 10, 1000)], dx=0.01)

    bayesian_theta = np.trapz(
        [theta * posterior(theta) / normalization_constant for theta in np.linspace(-10, 10, 1000)], dx=0.01)

    return bayesian_theta


true_theta = 2
mu = 0
sigma = 1
n_samples = 50

samples = normal_sample(n_samples, true_theta, sigma)

mu0 = 0
sigma0 = 1

bayesian_theta = bayesian_estimate(samples, mu0, sigma0)

print("Байесовская оценка параметра:", bayesian_theta)
