import click
from mlcost import mlcost
from mlcost import graphs


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass

@main.command()
# argparse.ArgumentParser(description='Dashboard to evaluate the cost of different ML models')
@click.option('-d', '--dataset', help='filepath to dataset, uses Iris dataset if none given', type=click.Path(exists=True, dir_okay=False))
@click.option('-l', '--labels', help='labels column, defaults to last column')
@click.option('-t', '--test', help='filepath to testset, if none will get split testset from dataset', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--separator', default=",", help='separator, defaults to a comma (no spaces)')
@click.option('-f', '--codecarbon-file', help="filename for default output from codecarbon, can be used with codecarbon's own visualization tool")
@click.option('--online', is_flag='True', help='use Codecarbon Emission Tracker in online mode')
@click.option('--log', is_flag='True', help='output to additional csv file with whole experiment data')
@click.option('--no-header', is_flag='True', help='do not consider the first row in the data to be the header')
def measure(dataset, labels, test, separator, codecarbon_file, online, log, no_header):
    mlcost.main(dataset, labels, test, separator, codecarbon_file, online, log, no_header)


@main.command()
# parser = argparse.ArgumentParser(description='Tool to graph the data obtained from the main app')
@click.option('-f', '--file', help='filepath to csv file that contains the data', required=True, type=click.Path(dir_okay=False))
def show(file):
    graphs.main(file)



if __name__ == "__main__":
    main()