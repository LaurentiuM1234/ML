from read_data import read_data
from scatter_matrix import scatter_matrix
from get_opt_vector import get_opt_vector
from plot_data import plot_data
import numpy as np
import matplotlib.pyplot as plt


def lda(p1_file, p2_file, file_delimiter=',', display_opt_v=True):
    # Reading the population matrices
    population_1, population_2 = read_data(p1_file, p2_file, file_delimiter)

    # Computing the scatter matrices
    p1_scatter = scatter_matrix(population_1)
    p2_scatter = scatter_matrix(population_2)

    # Computing the mean vectors
    p1_attributes, p1_samples = population_1.shape
    p2_attributes, p2_samples = population_2.shape

    p1_mean = np.reshape(population_1.mean(1), (p1_attributes, 1))
    p2_mean = np.reshape(population_2.mean(1), (p2_attributes, 1))

    # Computing the optimization vector
    opt_v = get_opt_vector(p1_scatter, p2_scatter, p1_mean, p2_mean)

    # Computing the projection matrix
    proj_matrix = np.matmul(opt_v, opt_v.T)

    # Plotting initial data
    plt.suptitle('LDA')
    plt.subplot(1, 2, 1)
    plot_data(population_1, population_2, opt_v, 'Initial data', display_opt_v)

    # Computing the projected data matrix
    proj_p1_matrix = np.matmul(proj_matrix, population_1)
    proj_p2_matrix = np.matmul(proj_matrix, population_2)

    # Plotting the projected data
    plt.subplot(1, 2, 2)
    plot_data(proj_p1_matrix, proj_p2_matrix, opt_v, 'Projected data', display_opt_v)

    plt.show()


if __name__ == '__main__':
    lda('/Users/laurentiumihalcea/Desktop/MLStuff/Data/p1.csv',
        '/Users/laurentiumihalcea/Desktop/MLStuff/Data/p2.csv')
else:
    print("Please run main file!")
