import numpy as np
import matplotlib.pyplot as plt
import pandas

from mlcost.learn import MODEL_TYPES


MARKERS = ['o', 'x', '+']
MARKER_LABELS = ['Low', 'Mid', 'High']

MEASUREMENTS = 10
LOADS = len(MARKERS)
N_MODELS = len(MODEL_TYPES)


def plot_accuracy_emissions(data: pandas.DataFrame, name):
    fig, ax = plt.subplots()
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
    plt.show()


def boxplot(data: pandas.DataFrame, name):
    fig, ax = plt.subplots()

    emissions = data.filter(like="emissions").to_numpy().T
    emissions = np.reshape(emissions, (LOADS * N_MODELS, MEASUREMENTS))

    ax.boxplot(emissions.T)
    ax.set_yscale("log")
 
    ax.set_title(name)
    plt.show()


def main(file):
    data = pandas.read_csv(file)
    name = data.iloc[0,0].split('/')[-1].split('.')[0].capitalize()
    plot_accuracy_emissions(data, name)
    boxplot(data, name)


if __name__== "__main__":
    main()
