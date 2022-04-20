import argparse
import logging

from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from codecarbon.emissions_tracker import BaseEmissionsTracker

import utils
import learn

def parse_args():
    """Parse the command-line arguments and return an argument object."""
    parser = argparse.ArgumentParser(description='Dashboard to evaluate the cost of different ML models')
    parser.add_argument('-d', '--dataset', help='filepath to dataset, uses Iris dataset if none given')
    parser.add_argument('-l', '--labels', help='labels column, defaults to last column')
    parser.add_argument('-t', '--test', help='filepath to testset, if none will get split testset from dataset')
    parser.add_argument('-s', '--separator', default=",", help='separator')
    parser.add_argument('-f', '--codecarbon-file', help="filename for default output from codecarbon, can be used with codecarbon's own visualization tool")
    parser.add_argument('--online', action='store_true', help='use Codecarbon Emission Tracker in online mode')
    parser.add_argument('--log', action='store_true', help='output to additional csv file with whole experiment data')
    parser.add_argument('--no-header', action='store_true', help='do not consider the first row in the data to be the header')
    return parser.parse_args()


def start_benchmark(online=False, save_to_file=False, name="") -> BaseEmissionsTracker:
    """Start emission and time tracking."""
    if online:
        tracker = EmissionsTracker(project_name=name, save_to_file=save_to_file)
    else:
        tracker = OfflineEmissionsTracker(country_iso_code="ESP" , project_name=name, save_to_file=save_to_file)

    tracker.persistence_objs.append(utils.EmissionsOutput())
    tracker.start()
    return tracker


def stop_benchmark(em_tracker: BaseEmissionsTracker):
    """Stop emission and time tracking."""
    em_tracker.stop()
    for output in em_tracker.persistence_objs:
        if isinstance(output, utils.EmissionsOutput):
            return output.data
    

def main():
    logging.getLogger('codecarbon').setLevel(logging.ERROR)
    utils.print_computer_info()

    args = parse_args()
    trainer = learn.Trainer(args.dataset, args.test, args.labels, args.separator, args.no_header, null_values='?')
    trainer.clean_data(log_output=True)
    
    try:
        for name, model in learn.MODEL_TYPES.items():
            filename = "_".join([args.codecarbon_file, name]) if args.codecarbon_file else ""
            em_tracker = start_benchmark(args.online, bool(args.codecarbon_file), filename)
            predictions = trainer.train(model)
            emission = stop_benchmark(em_tracker)
            score = trainer.score(predictions)

            utils.print_output(name, trainer.report, emission.duration, emission.emissions, emission.energy_consumed)

            if args.log:
                utils.log_to_file(trainer.name, score, emission, name)
    except KeyboardInterrupt:
        pass


if __name__== "__main__":
    main()
