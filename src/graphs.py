import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas

from learn import MODEL_TYPES


MARKERS = ['o', 'x', '+']
MARKER_LABELS = ['Low', 'Mid', 'High']


def parse_args():
    """Parse the command-line arguments and return an argument object."""
    parser = argparse.ArgumentParser(description='Tool to graph the data obtained from the main app')
    parser.add_argument('-f', '--file', help='filepath to csv file that contains the data', required=True)
    return parser.parse_args()


def plot_accuracy_emissions(data: pandas.DataFrame, name):
    fig, ax = plt.subplots()
    scatter_mat = []
    for i, model in enumerate(MODEL_TYPES):
        scatters = []
        for j in range(3):
            s = ax.scatter(model + "_emissions", model + "_accuracy", 
                data=data.iloc[10*j:10*(j+1)], 
                label=model, 
                color='C' + str(i),
                marker=MARKERS[j])
            scatters.append(s)
        scatter_mat.append(scatters)

    legend1 = ax.legend([l[0] for l in scatter_mat], MODEL_TYPES, loc='lower right', title="Models")
    ax.add_artist(legend1)
    ax.legend(scatter_mat[0], MARKER_LABELS, loc='lower right', bbox_to_anchor=(0.75, 0), title="CPU load")
    ax.set_title(name)
    ax.set_xlabel("Emissions")
    ax.set_ylabel("Accuracy")
    plt.show()


def main():
    args = parse_args()
    data = pandas.read_csv(args.file)
    name = data.iloc[0,0].split('/')[-1].split('.')[0].capitalize()
    plot_accuracy_emissions(data, name)


if __name__== "__main__":
    main()
