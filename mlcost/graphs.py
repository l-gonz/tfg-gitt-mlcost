import numpy as np
import matplotlib.pyplot as plt
import pandas

from mlcost.learn import MODEL_TYPES


MARKERS = ['o', 'x', '+', 'v', '^', "1", "s", "P", "*"]
MARKER_LABELS = ['Low', 'Mid', 'High']

MEASUREMENTS = 10
LOADS = len(MARKERS)
N_MODELS = len(MODEL_TYPES)


def plot_accuracy_emissions(data: pandas.DataFrame, name):
    _, ax = plt.subplots()
    scatter_mat = []
    for i, model in enumerate(MODEL_TYPES):
        scatters = []
        for j in range(LOADS):
            s = ax.scatter("emissions", "accuracy", 
                data=data[data['model'] == model].iloc[MEASUREMENTS*j:MEASUREMENTS*(j+1)], 
                label=model, 
                color='C' + str(i),
                marker=MARKERS[j])
            scatters.append(s)
        scatter_mat.append(scatters)

    legend1 = ax.legend([l[0] for l in scatter_mat], MODEL_TYPES, loc='lower right', title="Models")
    ax.add_artist(legend1)
    ax.legend(scatter_mat[0], MARKER_LABELS, loc='lower right', bbox_to_anchor=(0.75, 0), title="CPU load")
    ax.set_title(name)
    ax.set_xscale("log")
    ax.set_xlabel("Emissions")
    ax.set_ylabel("Accuracy")
    plt.show(block=False)


def boxplot(data: pandas.DataFrame, name):
    fig, ax = plt.subplots()

    emissions = data.filter(like="emissions").to_numpy().T
    emissions = np.reshape(emissions, (LOADS * N_MODELS, MEASUREMENTS))

    ax.boxplot(emissions.T)
    ax.set_yscale("log")
 
    ax.set_title(name)
    plt.show(block=False)

def scatter_four_dimensions(data, x_axis, y_axis, main_category, sec_category,
                            xscale="linear", xlabel=None, ylabel=None):
    main_cat_names = data[main_category].unique()
    sec_cat_names = data[sec_category].unique()
    scatter_mat = []
    _, ax = plt.subplots()

    for i, main in enumerate(main_cat_names):
        scatters = []
        for j, sec in enumerate(sec_cat_names):
            s = ax.scatter(x_axis, y_axis,
                data=data[(data[main_category] == main) & (data[sec_category] == sec)],
                label=main,
                color='C' + str(i),
                marker=MARKERS[j])
            scatters.append(s)
        scatter_mat.append(scatters)

    legend1 = ax.legend([l[0] for l in scatter_mat], main_cat_names, loc='lower right', title=main_category)
    ax.add_artist(legend1)
    ax.legend(scatter_mat[0], sec_cat_names, loc='lower right', bbox_to_anchor=(0.8, 0), title=sec_category)
    # ax.set_title(title)
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel if xlabel else x_axis)
    ax.set_ylabel(ylabel if ylabel else y_axis)
    plt.show(block=False)


def scatter_three_dimensions(data, x_axis, y_axis, category):
    category_names = data[category].unique()
    _, ax = plt.subplots()

    scatters = []
    for i, main in enumerate(category_names):
        s = ax.scatter(x_axis, y_axis,
            data=data[(data[category] == main)],
            label=main,
            color='C' + str(i))
        scatters.append(s)

    ax.legend(loc='lower right')
    ax.set_title("Plot")
    ax.set_xscale("log")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    plt.show(block=False)


def plot_bars(data: pandas.DataFrame, x_axis, y_axis, category, xcale="linear", yscale="linear"):
    ax = data.pivot(index=x_axis, columns=category, values=y_axis).plot(kind='bar', rot=0)
    # ax.set_ylim(0.5, 1)
    ax.set_title("Plot")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_xscale(xcale)
    ax.set_yscale(yscale)
    plt.show(block=False)

def plot_lines(data: pandas.DataFrame, x_axis, y_axis, category, 
               xscale="linear", yscale="linear",
               xlabel=None, ylabel=None, outside_legend=False):
    data_ref = data.pivot(index=x_axis, columns=category, values=y_axis)
    ax = data_ref.plot(marker='o')
    # ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else x_axis)
    ax.set_ylabel(ylabel if ylabel is not None else y_axis)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if outside_legend:
        box = ax.get_position() 
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title=category)
    ax.set_xticks(data_ref.index.codes)
    ax.set_xticklabels(data_ref.index.array)
    plt.show(block=False)


def __add_derived_columns(data):
    data["fit_emissions"] = data["emissions"] / data["emission_time"] * data["fit_time"]
    data["emissions_per_second"] = data["emissions"] / data["emission_time"]

def include(matrix, column, *names):
    return matrix[matrix[column].isin(names)]

def exclude(matrix, column, *names):
    return matrix[~matrix[column].isin(names)]


def main(file):
    data = pandas.read_csv(file, dtype={
        'model': pandas.CategoricalDtype(MODEL_TYPES.keys(), ordered = True), 
        "dataset": pandas.CategoricalDtype(ordered=True)})
    __add_derived_columns(data)
    average_data = data.groupby(["model", "dataset"], as_index=False, observed=True).mean()

    # ------- TEST --------
    # scatter_four_dimensions(average_data, "fit_emissions", "test_f1_score", "model", "dataset", "log")
    # scatter_three_dimensions(data[data["dataset"] == "Iris"], "fit_emissions", "test_f1_score", "model")
    # scatter_three_dimensions(data[data["dataset"] == "Ionosphere"], "fit_emissions", "test_f1_score", "model")
    # scatter_three_dimensions(data[data["dataset"] == "Banknote"], "fit_emissions", "test_f1_score", "model")
    # plot_bars(average_data, "model", "test_f1_score", "dataset")
    # plot_bars(average_data, "dataset", "test_f1_score", "model")

    # ------ Part 1 -------
    # Run with out/all-models-laptop.csv
    if "laptop" in file:
        scatter_four_dimensions(data, "fit_emissions", "test_f1_score", "model", "dataset", "log", 
                                "Emisiones por iteración [kg CO2eq]", "Valor-F [-]")
        plot_lines(average_data, "n_samples", "fit_emissions", "model", "log", "log",
                "Nº de muestras", "Emisiones medias por iteración [kg CO2eq]")
        plot_lines(average_data, "n_samples", "test_f1_score", "model", "log", "linear", 
                "Nº de muestras", "Valor-F medio [-]")

    # ------- Part 2 -------
    # Run with out/output-parallel-azure-electricity.csv
    if "parallel" in file:
        scatter_four_dimensions(data, "fit_emissions", "test_f1_score", "model", "dataset", "log",
                                "Emisiones por iteración [kg CO2eq]", "Valor-F [-]")
        scatter_four_dimensions(average_data, "fit_emissions", "test_f1_score", "model", "dataset", "log",
                                "Emisiones medias por iteración [kg CO2eq]", "Valor-F medio [-]")
        plot_lines(average_data, "dataset", "emissions_per_second", "model", yscale="log",
                xlabel="", ylabel="Emisiones por segundo [kg CO2eq / s]")
        plot_lines(average_data, "dataset", "emissions", "model", yscale="log",
                xlabel="", ylabel="Emisiones totales [kg CO2eq]", outside_legend=True)

    key_pressed = False
    while not key_pressed:
        key_pressed = plt.waitforbuttonpress()


if __name__== "__main__":
    main()
