import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, save, output_file, ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.palettes import Oranges8, BrBG8, BrBG11
import fiona
from math import pi


def find_nearest(array, val):
    """
    Function to find index of nearest grid-cell in array.
    """

    idx = np.abs(array - val).argmin()
    return idx


def set_colrange(data, breaks, type='minmax'):
    """
    Returns integers on a constant scale given low and high value from data.
    :param data: data to base range from
    :param breaks: number of breaks
    :param type: 'minmax'=>range is set by min and max of data; 'equal'=>range
     is set from the absolute max and the negative of the absolute max. This
     should be used for precipitation plots.
    """
    if type == 'minmax':
        r = np.linspace(data.min(), data.max(), breaks)
    elif type == 'equal':
        high = data.abs().max()
        r = np.linspace(-high, high, breaks)
    nd = []
    for val in data:
        nd.append(find_nearest(r, val))
    return nd


def add_colorbar(palette, low, high, plot_height=400):
    """
    Returns colorbar legend for bokeh plot
    palette - list of colors
    low - low data value
    high - high data value
    plot_height - value for height of plot
    """
    y = np.linspace(low, high, len(palette))
    dy = y[1] - y[0]
    legend = figure(tools="", x_range=[0, 1], y_range=[low-0.5*dy, high+0.5*dy],
                    plot_width=100, plot_height=plot_height, y_axis_location='right')
    legend.toolbar_location = None
    legend.xaxis.visible = False
    legend.rect(x=0.5, y=y, color=palette, width=1, height=dy)
    return legend


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


def clim_div_ann(clim_div_shp, stats_df, r_data, title="",
                 savepath="trash.html", stat='median', var='temp',
                 browser=True):
    """
    Returns change in temp. variable for each climate division.

    clim_div_shp - climate division shapefile (includes '.shp')
    stats_df - results from macastats.zstats()
    r_data - results from macastats.zstats_range()
    title - main figure title
    savepath - add path to save html file
    stat - stat to include from macastats.zstats() output
    var - 'temp' or 'precip'
    browser - should I show in the browser
    """

    # Get coordinates
    c = fiona.open(clim_div_shp+".shp")
    data = list(c)
    c.close()
    coords = [div['geometry']['coordinates'] for div in data]
    coords_clean = [div[0] for div in coords]
    xs = [[x for x, y in n] for n in coords_clean]
    ys = [[y for x, y in n] for n in coords_clean]
    cd_sorter = [d['properties']['CLIMDIV'] for d in data]  # Order of Climate Divisions

    # Make sure data is consistently sorted
    stats_df.climdiv = stats_df.climdiv.astype("category")
    stats_df.climdiv.cat.set_categories(cd_sorter, inplace=True)
    stats_df = stats_df.sort_values('climdiv')
    r_data.climdiv = r_data.climdiv.astype("category")
    r_data.climdiv.cat.set_categories(cd_sorter, inplace=True)
    r_data = r_data.sort_values('climdiv')

    # Set values
    values = list(stats_df[stat])
    cd_dict = {
        '2401': 'N West',
        '2402': 'S West',
        '2403': 'N Central',
        '2404': 'Central',
        '2405': 'S Central',
        '2406': 'N East',
        '2407': 'S East'
    }

    # Create colormap
    if var == 'temp':
        col_samp = Oranges8[::-1]  # reverse the order
        int_vals = set_colrange(stats_df[stat], 8, type='minmax')
        legend = add_colorbar(col_samp, stats_df[stat].min(), stats_df[stat].max(),
                              plot_height=700)
        webtitle = 'MT Change in Annual Temp.'
    elif var == 'precip':
        col_samp = BrBG11[::-1]  # reverse the order
        int_vals = set_colrange(stats_df[stat], 11, type='equal')
        legend = add_colorbar(col_samp, -stats_df[stat].abs().max(),
                              stats_df[stat].abs().max(), plot_height=700)
        webtitle = 'MT Change in Annual Precip.'

    colors = [col_samp[int(value)] for value in int_vals]
    climdivs = [cd_dict[str(cd)] for cd in stats_df['climdiv']]

    output_file(savepath, title=webtitle)

    source = ColumnDataSource(data=dict(
        climdiv=climdivs,
        x=xs,
        y=ys,
        value=values,
        min_val=r_data['min'],
        min_mod=r_data['model_min'],
        max_val=r_data['max'],
        max_mod=r_data['model_max'],
        perc_agree=r_data['perc_agree'],
        color=colors)
    )

    TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
    p = figure(tools=TOOLS, plot_width=1100, plot_height=700, title=title)
    p.grid.grid_line_color = None
    p.xaxis.axis_label = "Longitude"
    p.yaxis.axis_label = "Latitude"
    p.patches('x', 'y', line_color='white', line_width=1.5,
              source=source, fill_color='color')

    hover = p.select_one(HoverTool)
    hover.point_policy = 'follow_mouse'
    hover.tooltips = [
        ("Climate Div.", "@climdiv"),
        ("(Long, Lat)", "($x, $y)"),
        ("Ensemble Mean", "@value"),
        ('(Ensemble Min., Model)', '(@min_val, @min_mod)'),
        ('(Ensemble Max., Model)', '(@max_val, @max_mod)'),
        ('Model Agreement', '@perc_agree %')
    ]

    if browser:
        show(gridplot(p, legend, ncols=2))
    else:
        save(gridplot(p, legend, ncols=2))


def clim_div_grid(stats_df, stat='median', title='', r_data=None, browser=True,
                  save_path="./misc.html", var='temp'):
    """
    Plots change in temperature by month and climate division.
    stats_df - dataframe with zonal stats
    stat - column in stats_df to use as values
    title - title for plot
    browser - should open plot in browser?
    var - which variable are you plotting?
    r_data - dataframe with min and max of zonal stats
    save_path - where do you want to save the html file?
    """

    mth_samp = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                "Oct", "Nov", "Dec"]

    cd_dict = {
        '2401': 'N West',
        '2402': 'S West',
        '2403': 'N Central',
        '2404': 'Central',
        '2405': 'S Central',
        '2406': 'N East',
        '2407': 'S East'
    }

    # Make sure data is consistently sorted
    stats_df = stats_df.sort_values('climdiv')
    r_data = r_data.sort_values('climdiv')

    # Setup colorbar
    if var == 'temp':
        col_samp = Oranges8[::-1]  # reverse the order
        int_vals = set_colrange(stats_df[stat], 8, type='minmax')
        legend = add_colorbar(col_samp, stats_df[stat].min(), stats_df[stat].max())
        webtitle = 'MT Change in Monthly Temp.'
    elif var == 'precip':
        col_samp = BrBG8[::-1]  # reverse the order
        int_vals = set_colrange(stats_df[stat], 8, type='equal')
        legend = add_colorbar(col_samp, -stats_df[stat].abs().max(),
                              stats_df[stat].abs().max())
        webtitle = 'MT Change in Monthly Precip.'

    colors = [col_samp[val] for val in int_vals]
    months = [mth_samp[mth-1] for mth in stats_df['month']]
    climdivs = [cd_dict[str(cd)] for cd in stats_df['climdiv']]

    output_file(save_path, title=webtitle)

    source = ColumnDataSource(data=dict(
            climdiv=climdivs,
            month=months,
            value=stats_df[stat],
            min_val=r_data['min'],
            min_mod=r_data['model_min'],
            max_val=r_data['max'],
            max_mod=r_data['model_max'],
            perc_agree=r_data['perc_agree'],
            color=colors)
    )

    TOOLS = "hover,save"
    p = figure(title=title, x_range=mth_samp, y_range=cd_dict.values(),
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

    if r_data is None:
        p.select_one(HoverTool).tooltips = [
            ('Climate Division', '@climdiv'),
            ('Ensemble Mean', '@value')
        ]
    else:
        p.select_one(HoverTool).tooltips = [
            ('Month', '@month'),
            ('Climate Division', '@climdiv'),
            ('Ensemble Mean', '@value'),
            ('(Ensemble Min., Model)', '(@min_val, @min_mod)'),
            ('(Ensemble Max., Model)', '(@max_val, @max_mod)'),
            ('Model Agreement', '@perc_agree %')
        ]

    if browser:
        show(gridplot(p, legend, ncols=2))
    else:
        save(gridplot(p, legend, ncols=2))
