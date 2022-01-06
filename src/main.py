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
    parser.add_argument('--online', action='store_true', help='use Codecarbon Emission Tracker in online mode')
    parser.add_argument('--file', action='store_true', help='log model scores and emissions to csv file')
    return parser.parse_args()


def start_benchmark(online=False) -> BaseEmissionsTracker:
    """Start emission and time tracking."""
    if online:
        tracker = EmissionsTracker(project_name="ml-dashboard", save_to_file=False)
    else:
        tracker = OfflineEmissionsTracker(country_iso_code="ESP" , project_name="ml-dashboard", save_to_file=False)

    tracker.persistence_objs.append(utils.EmissionsOutput())
    tracker.start()
    return tracker


def stop_benchmark(em_tracker: BaseEmissionsTracker):
    """Stop emission and time tracking."""
    em_tracker.stop()
    return em_tracker.persistence_objs[0].data
    

def main():
    logging.getLogger('codecarbon').setLevel(logging.ERROR)
    utils.print_computer_info()

    args = parse_args()
    X_train, X_test, y_train, y_test = learn.get_sets(args.dataset, args.test, args.labels, args.separator)
    X_train_clean, X_test_clean = learn.clean_features(X_train, X_test)
    scores, emissions = {}, {}
    for name, model in learn.MODEL_TYPES.items():
        em_tracker = start_benchmark(args.online)
        model.fit(X_train_clean, y_train)
        predictions = model.predict(X_test_clean)
        emissions[name] = stop_benchmark(em_tracker)
        scores[name] = learn.get_score(y_test, predictions)
        utils.print_output(name, scores[name], emissions[name].duration, emissions[name].emissions, emissions[name].energy_consumed)

    if args.file:
        utils.log_to_file(args.dataset, scores, emissions, learn.MODEL_TYPES)


if __name__== "__main__":
    main()
