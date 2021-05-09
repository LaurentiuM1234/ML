import numpy as np
import matplotlib.pyplot as plt


def plot_data(p1_matrix, p2_matrix, opt_v, plot_title, plot_v = False):

    # Plotting population 1
    plt.plot(p1_matrix[0, :], p1_matrix[1, :], 'r+')

    # Plotting population 2
    plt.plot(p2_matrix[0, :], p2_matrix[1, :], 'b+')

    x_lim = np.max(np.abs(np.concatenate((p1_matrix[0, :], p2_matrix[0, :])))) + 10
    y_lim = np.max(np.abs(np.concatenate((p1_matrix[1, :], p2_matrix[1, :])))) + 10

    # Making QoL changes
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_label_coords(0.5, 0)
    ax.yaxis.set_label_coords(0, 0.5)
    ax.title.set_text(plot_title)
    plt.xlim((-x_lim, x_lim))
    plt.ylim((-y_lim, y_lim))
    plt.xlabel('Age')
    plt.ylabel('Height')

    # Plotting the opt. vector
    if plot_v:
        x = np.linspace(0, x_lim, 2)
        y = (opt_v[1, 0] / opt_v[0, 0]) * x
        plt.plot(x, y, 'g')

