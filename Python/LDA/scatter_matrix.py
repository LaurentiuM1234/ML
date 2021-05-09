import numpy as np


def scatter_matrix(p_matrix):
    # Function that calculates the scatter matrix
    # for a population

    # Initializing the scatter matrix
    attributes, samples = p_matrix.shape
    scatter = np.zeros((attributes, attributes))

    # Computing the average vector
    p_avg = p_matrix.mean(1)
    p_avg = np.reshape(p_avg, (attributes, 1))

    # Iterating over the samples
    for i in range(0, samples):
        c_samples = p_matrix[:, i] - p_avg
        scatter += np.matmul(c_samples, c_samples.T)

    return scatter




