import matplotlib.pyplot as plt
import numpy as np


def mod_diff_comp(precip, temp, mod_names=None):
    """
    Creates scatter plot of temp and precip difference for each model.
    :param precip: numpy array of projected change in precipitation for each model
    :param temp: numpy array of projected change in temp for each model
    :return: scatter plot
    """
    precip_avg = np.mean(precip, axis=1).mean(axis=1)
    temp_avg = np.mean(temp, axis=1).mean(axis=1)

    plt.scatter(precip_avg, temp_avg)
    plt.title("Projected Change from CMIP5 Models")
    plt.xlabel("Change in Annual Precipitation ()")
    plt.ylabel("Change in Annual Temperature ()")

    if mod_names:
        for label, x, y, in zip(mod_names, precip_avg, temp_avg):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round, pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0')
            )
    plt.show()
