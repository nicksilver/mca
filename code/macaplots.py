import matplotlib.pyplot as plt
import numpy as np


def mod_diff_comp(precip1, temp1, leg_lab1="RCP 4.5",
                  precip2=None, temp2=None, leg_lab2="RCP 8.5",
                  title="", mod_names=None, annotate=False):
    """
    Creates scatter plot of temp and precip1 difference for each model.
    :param precip: numpy array of projected change in precipitation for each model
    :param temp: numpy array of projected change in temp for each model
    :param leg_lab: str for legend label
    :param title: str for figure title
    :param mod_names: list of model names used to create legend and annotate plot
    :param annotate: should the markers be annotated?
    :return: scatter plot
    """
    precip_avg1 = np.mean(precip1, axis=1).mean(axis=1)
    temp_avg1 = np.mean(temp1, axis=1).mean(axis=1)

    p1 = plt.scatter(precip_avg1, temp_avg1, s=100, marker="^", edgecolors="k",
                     color="blue")

    if precip2 is not None:
        precip_avg2 = np.mean(precip2, axis=1).mean(axis=1)
        temp_avg2 = np.mean(temp2, axis=1).mean(axis=1)
        p2 = plt.scatter(precip_avg2, temp_avg2, s=100, marker="^",
                         edgecolors="k", color="red")
    plt.legend([p1, p2], [leg_lab1, leg_lab2])
    plt.title(title)
    plt.xlabel("Change in Annual Precipitation (mm)")
    plt.ylabel("Change in Annual Temperature (C)")

    if annotate:
        for label, x, y, in zip(mod_names, precip_avg1, temp_avg1):
            plt.annotate(
                label,
                xy=(x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom',
                #bbox=dict(boxstyle='round, pad=0.5', fc='yellow', alpha=0.5),
                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0')
            )
    for label, x, y, in zip(mod_names, precip_avg2, temp_avg2):
        plt.annotate(
            label,
            xy=(x, y), xytext=(0, 10),
            textcoords='offset points', ha='center', va='bottom',
            #bbox=dict(boxstyle='round, pad=0.5', fc='yellow', alpha=0.5),
            #arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0')
        )
    plt.show()
