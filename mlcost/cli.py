import click
from mlcost import mlcost
from mlcost import graphs


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass

@main.command()
# argparse.ArgumentParser(description='Dashboard to evaluate the cost of different ML models')
@click.option('-d', '--dataset', help='filepath to dataset, uses Iris dataset if none given. If using the --openml option, it is the dataset id in openml', type=click.Path(exists=False, dir_okay=False))
@click.option('-l', '--labels', help='labels column, defaults to last column')
@click.option('-t', '--test', help='filepath to test set, if none will get split test set from dataset', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--separator', default=",", help='separator, defaults to a comma (no spaces)')
@click.option('-f', '--codecarbon-file', help="filename for default output from codecarbon, can be used with codecarbon's own visualization tool")
@click.option('-m', '--model', help="specify one model to run, default is to run everything")
@click.option('-cv', '--cross-validate', help="cross validate the model, specify the number of folds (default is 1, no cross validation)", type=int, default=1)
@click.option('--online', is_flag='True', help='use Codecarbon Emission Tracker in online mode')
@click.option('--log', is_flag='True', help='output to additional csv file with whole experiment data')
@click.option('--no-header', is_flag='True', help='do not consider the first row in the data to be the header')
@click.option('--openml', is_flag='True', help='fetch dataset from openml')
@click.option('--parallel', is_flag='True', help='parellelize cross validation between all cores, needs -cv option')
def measure(dataset, labels, test, separator, codecarbon_file, model, cross_validate, online, log, no_header, openml, parallel):
    mlcost.main(dataset, labels, test, separator, codecarbon_file, model, cross_validate, online, log, no_header, openml, parallel)


@main.command()
# parser = argparse.ArgumentParser(description='Tool to graph the data obtained from the main app')
@click.option('-f', '--file', help='filepath to csv file that contains the data', required=True, type=click.Path(dir_okay=False))
def plot(file):
    graphs.main(file)



if __name__ == "__main__":
    main()