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

def scatter_four_dimensions(data, y_axis, x_axis, main_category, sec_category, xscale="linear"):
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
    ax.legend(scatter_mat[0], sec_cat_names, loc='lower right', bbox_to_anchor=(0.75, 0), title=sec_category)
    ax.set_title("Plot")
    ax.set_xscale(xscale)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    plt.show(block=False)


def scatter_three_dimensions(data, y_axis, x_axis, category):
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


def plot_bars(data: pandas.DataFrame, y_axis, x_axis, category):
    ax = data.pivot(index=x_axis, columns=category, values=y_axis).plot(kind='bar', rot=0)
    ax.set_ylim(0.5, 1)
    ax.set_title("Plot")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    plt.show(block=False)

def plot_lines(data: pandas.DataFrame, y_axis, x_axis, category, xcale="linear", yscale="linear"):
    ax = data.pivot(index=x_axis, columns=category, values=y_axis).plot(marker='o')
    ax.set_title("Plot")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_xscale(xcale)
    ax.set_yscale(yscale)
    plt.show(block=False)


def __add_derived_columns(data):
    data["fit_emissions"] = data["emissions"] / data["emission_time"] * data["fit_time"]


def main(file):
    data = pandas.read_csv(file)
    __add_derived_columns(data)
    average_data = data.groupby(["model", "dataset"], as_index=False).mean()

    # name = data.iloc[0,0].split('/')[-1].split('.')[0].capitalize()
    # plot_accuracy_emissions(data, name)
    # boxplot(data, name)

    
    # scatter_four_dimensions(average_data, "test_f1_score", "fit_emissions", "model", "dataset", "log")
    # scatter_three_dimensions(data[data["dataset"] == "Iris"], "test_f1_score", "fit_emissions", "model")
    # scatter_three_dimensions(data[data["dataset"] == "Ionosphere"], "test_f1_score", "fit_emissions", "model")
    # scatter_three_dimensions(data[data["dataset"] == "Banknote"], "test_f1_score", "fit_emissions", "model")
    # plot_bars(average_data, "test_f1_score", "model", "dataset")
    # plot_bars(average_data, "test_f1_score", "dataset", "model")

    # ------
    # plot_lines(average_data[average_data["dataset"] != "Optdigits"], "fit_emissions", "n_samples", "model", "log", "log")
    # plot_lines(average_data[average_data["dataset"] != "Optdigits"], "test_f1_score", "n_samples", "model")
    # scatter_four_dimensions(data, "test_f1_score", "fit_emissions", "model", "dataset", "log")
    # scatter_four_dimensions(data, "test_f1_score", "fit_emissions", "dataset", "model", "log")

    key_pressed = False
    while not key_pressed:
        key_pressed = plt.waitforbuttonpress()


if __name__== "__main__":
    main()
