import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.palettes import Oranges8
import fiona
from math import pi


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


def mod_diff_comp_bok(precip1, temp1, mod_names, filepath=None,
                  precip2=None, temp2=None, title=""):
    """
    Creates an INTERACTIVE html scatter plot of temp and precip difference for each
    specified model.
    :param precip: numpy array of projected change in precipitation for each model
    :param temp: numpy array of projected change in temp for each model
    :param mod_names: list of model names used to create legend and annotate plot
    :param title: str for figure title
    :return: scatter plot
    """

    # Flatten numpy array
    precip_avg1 = np.mean(precip1, axis=1).mean(axis=1)
    temp_avg1 = np.mean(temp1, axis=1).mean(axis=1)

    if precip2 is None:

        # Save html file
        if filepath is not None:
            output_file(filepath)

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

        # Open browser with image
        show(p)

    elif precip2 is not None:

        # Save html file
        if filepath is not None:
            output_file(filepath)

        # Flatten numpy array and add to source data
        precip_avg2 = np.mean(precip2, axis=1).mean(axis=1)
        temp_avg2 = np.mean(temp2, axis=1).mean(axis=1)
        source = ColumnDataSource(
            data=dict(
                prec1=list(precip_avg1),
                prec2=list(precip_avg2),
                temp1=list(temp_avg1),
                temp2=list(temp_avg2),
                model=mod_names
            )
        )

        TOOLS = "pan,wheel_zoom,save"
        p = figure(tools=TOOLS, title=title)
        tips1 = [
            ("Scenario", "RCP 4.5"),
            ("Model", "@model"),
            ("Delta precip.", "@prec1"),
            ("Delta temp.", "@temp1")
            ]
        tips2 = [
            ("Scenario", "RCP 8.5"),
            ("Model", "@model"),
            ("Delta precip.", "@prec2"),
            ("Delta temp.", "@temp2")
            ]

        r1 = p.triangle("prec1", "temp1", color="blue", size=15,
                       legend="RCP 4.5", source=source, fill_alpha=0.5)
        p.add_tools(HoverTool(renderers=[r1], tooltips=tips1))

        r2 = p.triangle("prec2", "temp2", color="red", size=15,
                       legend="RCP 8.5", source=source, fill_alpha=0.5)
        p.add_tools(HoverTool(renderers=[r2], tooltips=tips2))

        p.xaxis.axis_label = "Change in Annual Precipitation (mm/day)"
        p.yaxis.axis_label = "Change in Annual Temperature (C)"

        # Open browser with image
        show(p)


class clim_divs(object):
    """
    Object to plot climate data by climate divisions.
    """
    def __init__(self, clim_div_shp):
        self.clim_div_shp = clim_div_shp
        c = fiona.open(clim_div_shp)
        self.data = list(c)
        c.close()
        coords = [div['geometry']['coordinates'] for div in self.data]
        coords_clean = [div[0] for div in coords]
        self.xs = [[x for x, y in n] for n in coords_clean]
        self.ys = [[y for x, y in n] for n in coords_clean]


    def temp_plot(self, stats_df, title="", savepath=None, stat='mean'):
        """
        Returns change in temp. variable for each climate division.

        clim_div_shp - climate division shapefile (includes '.shp')
        stats_df - results from macastats.zstats()
        title - main figure title
        savepath - add path to save html file
        stat - stat to include from macastats.zstats() output
        """

        values = list(stats_df[stat])
        cd_names = list(stats_df['climdiv'])
        colors = [Oranges8[int(value)] for value in values]

        source = ColumnDataSource(data=dict(
            x=self.xs,
            y=self.ys,
            color=colors,
            name=cd_names,
            difference=values))

        TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
        p = figure(tools=TOOLS, plot_width=1100, plot_height=700, title=title)
        p.grid.grid_line_color = None
        p.xaxis.axis_label = "Longitude"
        p.yaxis.axis_label = "Latitude"
        p.patches('x', 'y', fill_alpha=0.5, line_color='white', line_width=1.5,
                  source=source, fill_color='color')

        hover = p.select_one(HoverTool)
        hover.point_policy = 'follow_mouse'
        hover.tooltips = [
            ("Climate Div.", "@name"),
            ("Difference", "@difference"),
            ("(Long, Lat)", "($x, $y)")
        ]

        if savepath is not None:
            output_file(savepath)
        show(p)

    def prec_plot(self, stats_dict, title="", savepath=None):
        """
        Returns change in precip variable for each climate division.

        clim_div_shp - climate division shapefile (includes '.shp')
        stats_dict - results from macastats.zstats()
        title - main figure title
        savepath - add path to save html file
        """
        pass

def force_range(data, maxi):
    """
    Returns data that spans the range from 0 to maxi
    maxi -- maximum value of the range
    """
    fact = np.float64(maxi)/(max(data - min(data)))
    nd = fact*(data-min(data))
    return list(nd.astype(int))


def clim_div_temp_grid(stats_df, stat='mean', title=''):
    """
    Returns change in temperature by month and climate division.
    """
    mth_samp = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                "Oct", "Nov", "Dec"]

    col_samp = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce",
                "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]

    cd_samp = ["N West", "S West", "N Central", "Central", "S Central", "N East",
                "S East"]

    int_vals = force_range(stats_df[stat], 8)
    colors = [col_samp[val] for val in int_vals]
    months = [mth_samp[mth-1] for mth in stats_df['month']]
    climdivs = [cd_samp[cd - 2401] for cd in stats_df['climdiv']]

    output_file('temp_grid.html', title="MT Clim Div Temperature")

    source = ColumnDataSource(data=dict(
            climdiv=climdivs,
            month=months,
            values=stats_df[stat],
            color=colors)
    )

    TOOLS = "hover,save,pan,box_zoom,wheel_zoom"
    p = figure(title=title, x_range=mth_samp, y_range=cd_samp,
               x_axis_location='above', plot_width=900, plot_height=400,
               tools=TOOLS)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.axis.major_label_text_font_size = "15pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi/3

    p.rect("month", "climdiv", 1, 1, source=source, color="color",
           line_color='black')

    p.select_one(HoverTool).tooltips = [
        ('Month', '@month'),
        ('Climate Division', '@climdiv'),
        ('Difference', '@values'),
    ]
    show(p)
