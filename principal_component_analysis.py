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
        data (np.ndarray): Dataset with rows as entries and columns as variables

    Returns:
        tuple[np.ndarray, np.ndarray]: Principal components directions and associated variance.
    """
    # Center the data
    data = center(data)

    # Find the covariance matrix
    cov = np.matmul(data.transpose(), data) / (data.shape[0] - 1)

    # Find the eigens
    evals, evects = np.linalg.eig(cov)
    idx = np.argsort(evals, axis=0)[::-1]
    sorted_eig_vectors = evects[:, idx]

    return evals, evects

# Testing -------------------------------------------
data = np.zeros((20,5))
data[:,0] = np.random.uniform(0, 10, 20)
data[:,1] = np.random.uniform(0,10, 20)
data[:,2] = data[:,0] + data[:,1] + np.random.normal(0, 1.15, 20)
data[:,3] = 9 * data[:,0] + 1 * data[:,1] + np.random.normal(0, 1.15, 20)
data[:,4] = 1 * data[:,0] + 9 * data[:,1] + np.random.normal(0, 1.15, 20) 

print(principal_components(data)[0])
print(principal_components(data)[1])