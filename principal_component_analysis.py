# Principal Component Tools

import numpy as np

def center(data: np.ndarray) -> np.ndarray:
    """ Centers the data for each column in an array

    Args:
        data (np.ndarray): The data to be centered. Each row is an entry of data, and each column represents a specific variable.

    Returns:
        np.ndarray: The center that is now centered. Same format as entry
    """
    # For each column, we need to find the mean and subtract that value
    for n in range(data.shape[1]):
        mean = np.mean(data[:,n])
        data[:,n] -= mean
    
    return data

def principal_components(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outputs the principal components of the data, along with the variance associated with each.

    Args:
        data (np.ndarray): Centered dataset with rows as entries and columns as variables

    Returns:
        tuple[np.ndarray, np.ndarray]: Principal components directions and associated variance.
    """
    # Find the covariance matrix
    cov = np.matmul(data.transpose(), data)

    # Find the eigens
    evals, evects = np.linalg.eig(cov)
    evals = np.flip(np.sort(evals))

    return evals, evects