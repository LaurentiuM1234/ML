import numpy as np


def read_data(p1_file, p2_file, file_delimiter=','):
    # Function that reads data from specified files
    # and returns the data matrices for the 2 pop.

    # Preparing the 2 population matrices
    population_1 = np.genfromtxt(p1_file, delimiter=file_delimiter)
    population_1 = population_1.transpose()

    population_2 = np.genfromtxt(p2_file, delimiter=file_delimiter)
    population_2 = population_2.transpose()

    return population_1, population_2


