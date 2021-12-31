import argparse
import logging
import timeit
import platform
import subprocess
import re
import os

import pandas as pd
from pandas.core.frame import DataFrame
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

MODEL_TYPES = {
    'Linear': RidgeClassifier(),
    'SupportVector': SVC(),
    'StochasticGradient': SGDClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(),
    'Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

FILE_NAME = 'output.csv'

INFO_COLUMN_NAMES = ["dataset", ""]
MODEL_COLUMN_NAMES = ["accuracy", "time", "emissions"]

def parse_args():
    """Parse the command-line arguments and return an argument object."""
    parser = argparse.ArgumentParser(description='Dashboard to evaluate the cost of different ML models')
    parser.add_argument('-d', '--dataset', help='filepath to dataset, uses Iris dataset if none given')
    parser.add_argument('-l', '--labels', help='labels column, defaults to last column')
    parser.add_argument('-t', '--test', help='filepath to testset, if none will get split testset from dataset')
    parser.add_argument('-s', '--separator', default=",", help='separator')
    parser.add_argument('--online', action='store_true', help='use Codecarbon Emission Tracker in online mode')
    return parser.parse_args()

def get_sets(args):
    """Return the train and test sets for the features and labels arrays."""
    if not args.dataset:
        X, y = load_iris(return_X_y=True, as_frame=True)
    else:
        data = pd.read_csv(args.dataset, sep=args.separator, engine='python')
        X, y = split_labels(data, args.labels)
        # Use separate table as test set if provided
        if args.test:
            test_data = pd.read_csv(args.dataset, sep=args.separator, engine='python')
            X_test, y_test = split_labels(test_data, args.labels)
            return X, X_test, y, y_test

    return train_test_split(X, y, train_size=0.8, test_size=0.2)

def split_labels(data: DataFrame, label_col: str):
    """Separate the features from the labels.
    
    Parameters
    ----------
    data -- a DataFrame containing features and labels
    label_col -- the name of the column with the labels
    """
    if not label_col:
        label_col = data.columns[-1]

    # Remove rows with missing labels
    data_all_labeled = data.dropna(axis=0, subset=[label_col])

    # Separate features and labels
    y = data_all_labeled[label_col]
    X = data_all_labeled.drop([label_col], axis=1)

    return X, y

def clean_features(X_train: DataFrame, X_test: DataFrame):
    """Return a clean version of the input data.
    
    Drop categorical features with more than ten categories and
    translates the others with One-Hot encoding.
    Fill missing numerical values with the mean.
    """
    # Separate numerical and categorical columns
    numerical_cols = [col for col in X_train.columns if 
                X_train[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X_train.columns if X_train[col].nunique() < 10 and 
                        X_train[col].dtype == "object"]

    # Make copy and drop rows with missing categories
    X_train_clean = X_train[numerical_cols + categorical_cols].copy()
    X_test_clean = X_test[numerical_cols + categorical_cols].copy()
    X_train_clean = X_train_clean.dropna(axis=0, subset=categorical_cols)
    X_test_clean = X_test_clean.dropna(axis=0, subset=categorical_cols)

    # Replace empty numerical items with mean and one-shot categorical columns
    numerical_transformer = SimpleImputer()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X_train_clean = preprocessor.fit_transform(X_train_clean)
    X_test_clean = preprocessor.transform(X_test_clean)

    return X_train_clean, X_test_clean

def start_benchmark(online=False):
    if online:
        tracker = EmissionsTracker(project_name="ml-dashboard", save_to_file=False)
    else:
        tracker = OfflineEmissionsTracker(country_iso_code="ESP" , project_name="ml-dashboard", save_to_file=False)
    tracker.start()
    time = timeit.default_timer()
    return tracker, time

def stop_benchmark(em_tracker, time_tracker):
    return em_tracker.stop(), timeit.default_timer() - time_tracker

def get_score(labels, predictions) -> int:
    return accuracy_score(labels, predictions)

def print_output(name, score, emissions, time):
    print("---------------------------")
    print(name)
    print(f'Accuracy: {score:.4f}')
    print(f"Emissions: {emissions:.4e}kg (CO2 equ)")
    print(f"Time: {time:.4f}s")

def print_computer_info():
    print("Running on " + platform.node())
    print(platform.freedesktop_os_release()['PRETTY_NAME'] + " " + platform.machine())
    print("Python " + platform.python_version())
    if platform.system() == "Linux":
        output = subprocess.check_output("cat /proc/cpuinfo", shell=True).strip().decode().split('\n')
        cpu_info = {item[0]: item[1] for item in [re.split("\s*:\s*", line, maxsplit=2) for line in output]}
        if 'model name' in cpu_info:
            print(cpu_info['model name'])
        output = subprocess.check_output("cat /proc/meminfo", shell=True).strip().decode().split('\n')
        mem_info = {item[0]: item[1] for item in [re.split("\s*:\s*", line, maxsplit=2) for line in output]}
        if 'MemTotal' in mem_info:
            ram = mem_info['MemTotal'].split()
            count = 0
            while int(ram[0]) >= 1024:
                count += 1
                ram[0] = int(ram[0]) / 1024
            print("Memory: " + str("%.2f" % round(ram[0],2)) + " " + ("kB" if count == 0 else "MB" if count == 1 else "GB"))

def get_column_names(model_name):
    return ",".join([model_name + "_" + column for column in MODEL_COLUMN_NAMES])

def log_to_file(dataset, scores, emissions, time):
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'w') as file:
            file.write(','.join(INFO_COLUMN_NAMES))
            file.write(','.join([get_column_names(name) for name in MODEL_TYPES.keys()]))

    with open(FILE_NAME, 'a') as file:
        file.write('\n')
        file.write(dataset if dataset else "iris" + ',')
        file.write(','.join([f"{scores[name]},{time[name]},{emissions[name]}" for name in MODEL_TYPES.keys()]))
    

def main():
    logging.getLogger('codecarbon').setLevel(logging.ERROR)
    print_computer_info()

    args = parse_args()
    X_train, X_test, y_train, y_test = get_sets(args)
    X_train_clean, X_test_clean = clean_features(X_train, X_test)
    scores, emissions, time = {}, {}, {}
    for name, model in MODEL_TYPES.items():
        em_tracker, time_tracker = start_benchmark(args.online)
        model.fit(X_train_clean, y_train)
        predictions = model.predict(X_test_clean)
        emissions[name], time[name] = stop_benchmark(em_tracker, time_tracker)
        scores[name] = get_score(y_test, predictions)
        print_output(name, scores[name], emissions[name], time[name])

    log_to_file(args.dataset, scores, emissions, time)


if __name__== "__main__":
    main()
