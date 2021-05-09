import numpy as np


def get_opt_vector(p1_scatter, p2_scatter, p1_mean, p2_mean):
    # Function that computes the optimization vector for the LDA
    # problem

    # Computing the sum of the scatter matrices
    sigma = p1_scatter + p2_scatter

    # Computing the inverse of the joint scatter matrix
    sigma_inv = np.linalg.inv(sigma)

    # Computing the optimization vector
    v = np.matmul(sigma_inv, (p1_mean - p2_mean))

    # Norming the optimization vector
    v = v / np.linalg.norm(v)

    return v
