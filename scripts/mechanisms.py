import numpy as np
from diffprivlib.mechanisms import GeometricTruncated

def krr(x, domain, epsilon):
    """k-Randomized Responde (kRR) mechanism.
    
    Parameters:
        x (any): Value to be randomised.
        domain (list): Domain.
        epsilon (float): Privacy parameter.

    Return:
        x' (any): Randomized value (a value from the domain).
    """
    # Probability to keep the original value
    domain_size = len(domain)
    p = np.exp(epsilon) / (np.exp(epsilon) + domain_size - 1)

    # Bernoulli experiment
    if np.random.binomial(n=1, p=p) == 1:
        return x

    # Delete the original value from the domain
    other_values = domain.copy()
    del other_values[other_values.index(x)]

    # Return a value different of the given value x
    return np.random.choice(other_values)

def geometric_truncated(x, lower, upper, epsilon):
    """Geometric truncated mechanism.
    
    Parameters:
        x (int): Value to be randomized.
        lower (int): Lower bound of the domain.
        upper (int): Upper bound of the domain.
        epsilon (float): Privacy parameter.

    Return:
        x' (int): A randomised value lower <= x'<= upper.
    """
    sensitivity = upper - lower
    M = GeometricTruncated(epsilon=epsilon, sensitivity=sensitivity, lower=lower, upper=upper)
    return M.randomise(x)
