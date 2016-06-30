import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import HoverTool
import pandas as pd


def mod_diff_comp(precip1, temp1, leg_lab1="RCP 4.5",
                  precip2=None, temp2=None, leg_lab2="RCP 8.5",
                  title="", mod_names=None, annotate=False,
                  id_subset=[]):
    """
    Creates scatter plot of temp and precip difference for each model.
    :param precip: numpy array of projected change in precipitation for each model
    :param temp: numpy array of projected change in temp for each model
    :param leg_lab: str for legend label
    :param title: str for figure title
    :param mod_names: list of model names used to create legend and annotate plot
    :param annotate: should the markers be annotated?
    :param id_subset: list of the model of names that are a subset of all mod_names
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

    id_idx = np.zeros(len(mod_names), dtype=bool)
    if len(id_subset) > 0:
        if mod_names is None:
            raise Exception("Need to provide a 'mod_names' parameter")
        for id in id_subset:
            id_idx[mod_names.index(id)] = True

        plt.scatter(precip_avg1[id_idx], temp_avg1[id_idx], s=300, marker="o",
                    facecolors='none', edgecolors="red")

        if precip2 is not None:
            plt.scatter(precip_avg2[id_idx], temp_avg2[id_idx], s=300, marker="o",
                        facecolors='none', edgecolors="red")

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
            )
        for label, x, y, in zip(mod_names, precip_avg2, temp_avg2):
            plt.annotate(
                label,
                xy=(x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom',
            )
    plt.show()


def mod_diff_comp_bok(precip1, temp1, mod_names, filepath=None, leg_lab1="RCP 4.5",
                  precip2=None, temp2=None, leg_lab2="RCP 8.5",
                  title="", annotate=False, id_subset=[]):
    """
    Creates an INTERACTIVE html scatter plot of temp and precip difference for each
    specified model.
    :param precip: numpy array of projected change in precipitation for each model
    :param temp: numpy array of projected change in temp for each model
    :param leg_lab: str for legend label
    :param title: str for figure title
    :param mod_names: list of model names used to create legend and annotate plot
    :param annotate: should the markers be annotated?
    :param id_subset: list of the model of names that are a subset of all mod_names
    :return: scatter plot
    """
    # Flatten numpy array
    precip_avg1 = np.mean(precip1, axis=1).mean(axis=1)
    temp_avg1 = np.mean(temp1, axis=1).mean(axis=1)

    if precip2 is None:

        # Configure Bokeh tools
        source = ColumnDataSource(
            data=dict(
                x=list(precip_avg1),
                y=list(temp_avg1),
                model=mod_names
            )
        )
        TOOLS = "pan,wheel_zoom,save,hover"
        p = figure(tools=TOOLS)

        # Create scatter plot
        p.scatter(precip_avg1, temp_avg1, size=15, source=source)

        # Custom dictionary for tooltip
        p.select_one(HoverTool).tooltips = [
            ("Model", "@model"),
            ("Delta Precip", "@x"),
            ("Delta Temp", "@y")
            ]

        # Save html file
        if filepath is not None:
            output_file(filepath)

        # Open browser with image
        show(p)

    elif precip2 is not None:

        # Flatten numpy array
        precip_avg2 = np.mean(precip2, axis=1).mean(axis=1)
        temp_avg2 = np.mean(temp2, axis=1).mean(axis=1)

        # Create pandas dataframe
        # prec = np.hstack((precip_avg1, precip_avg2))
        # temp = np.hstack((temp_avg1, temp_avg2))
        # rcp = np.hstack((np.repeat("RCP 4.5", len(precip_avg1)),
        #                  np.repeat("RCP 8.5", len(precip_avg2))))

        # Create colormap
        # colormap = {'RCP 4.5': 'blue', 'RCP 8.5': 'red'}
        # colors = [colormap[x] for x in list(rcp)]

        # source = ColumnDataSource(
        #     data=dict(
        #         prec=list(prec),
        #         temp=list(temp),
        #         rcp=list(rcp),
        #         model=mod_names+mod_names
        #     )
        # )

        #TODO Confirm that mod_names aligns with precip and temp

        TOOLS = "pan,wheel_zoom,save,hover"
        tips1 = [
            ("Scenario", "RCP 4.5"),
            ("Model", mod_names),
            ("Delta Precip", precip_avg1),
            ("Delta Temp", temp_avg1)
            ]
        tips2 = [
            ("Scenario", "RCP 8.5"),
            ("Model", mod_names),
            ("Delta Precip", precip_avg2),
            ("Delta Temp", temp_avg2)
            ]

        # p.select(HoverTool).tooltips = [
        #     ("Scenario", "@rcp"),
        #     ("Model", "@model"),
        #     ("Delta Precip", "@prec"),
        #     ("Delta Temp", "@temp")
        #     ]
        p = figure()

        # Create scatter plot
        # p.scatter(prec, temp, color=colors, size=15, source=source, legend=True)
        # p.legend
        r1 = p.scatter(precip_avg1, temp_avg1, color="blue", size=15, legend="RCP 4.5")
                       #source=source)
        p.add_tools(HoverTool(renderers=[r1], tooltips=tips1))
        r2 = p.scatter(precip_avg2, temp_avg2, color="red", size=15, legend="RCP 8.5")
                       #source=source)
        p.add_tools(HoverTool(renderers=[r2], tooltips=tips2))


        # Custom dictionary for tooltip
        # p.select(HoverTool).tooltips = [
        #     ("Scenario", "@rcp"),
        #     ("Model", "@model"),
        #     ("Delta Precip", "@prec"),
        #     ("Delta Temp", "@temp")
        #     ]

        # Save html file
        if filepath is not None:
            output_file(filepath)

        # Open browser with image
        show(p)






def clim_div_temp_grid():
    """
    Returns change in temperature by season and climate division.
    """
    pass
