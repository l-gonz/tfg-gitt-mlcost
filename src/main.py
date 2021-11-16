import argparse

import pandas as pd
from pandas._libs import missing
from pandas.core.frame import DataFrame

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

MODEL_TYPES = {
    'Forest': RandomForestClassifier()
}

def get_args():
    parser = argparse.ArgumentParser(description='Dashboard to evaluate the cost of different ML models')
    parser.add_argument('-d', '--dataset', help='filepath to dataset')
    parser.add_argument('-t', '--target', default='target', help='target column')
    parser.add_argument('--test', help='filepath to testset, if none will get split testset from dataset')
    parser.add_argument('-s', '--separator', help='separator')
    return parser.parse_args()

def get_sets():
    args = get_args()
    if args.dataset == None:
        X, y = load_iris(return_X_y=True, as_frame=True)
    else:
        data = pd.read_csv(args.dataset) if args.separator == None else pd.read_csv(args.dataset, sep=args.separator)
        X, y = clean_target(data, args.target)
    print(X.columns)
    if args.test == None or args.dataset == None:
        return train_test_split(X, y, train_size=0.8, test_size=0.2)
    else:
        test_data = pd.read_csv(args.test) if args.separator == None else pd.read_csv(args.dataset, sep=args.separator)
        X_test, y_test = clean_target(test_data, args.target)
        return X, X_test, y, y_test

def clean_target(data: DataFrame, target: str):
    # Remove rows with missing labels
    data_all_labeled = data.dropna(axis=0, subset=[target])

    # Ordinal-encode categories
    enc = OrdinalEncoder()
    data_all_labeled[[target]] = enc.fit_transform(data_all_labeled[[target]])

    # Separate features and labels
    y = data_all_labeled[target]
    X = data_all_labeled.drop([target], axis=1)

    return X, y

def clean_features(X_train, X_test):

    numerical_cols = [col for col in X_train.columns if 
                X_train[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X_train.columns if X_train[col].nunique() < 10 and 
                        X_train[col].dtype == "object"]

    X_train_clean = X_train[numerical_cols + categorical_cols].copy()
    X_test_clean = X_test[numerical_cols + categorical_cols].copy()
    X_train_clean = X_train_clean.dropna(axis=0, subset=categorical_cols)
    X_test_clean = X_test_clean.dropna(axis=0, subset=categorical_cols)

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

def start_benchmark():
    # Start codecarbon measures
    # Start timing measures
    pass

def stop_benchmark():
    pass

def get_score(target, predictions) -> int:
    print(classification_report(target, predictions))
    return accuracy_score(target, predictions)


X_train, X_test, y_train, y_test = get_sets()
X_train_clean, X_test_clean = clean_features(X_train, X_test)
scores = {}
for name, model in MODEL_TYPES.items():
    start_benchmark()
    model.fit(X_train_clean, y_train)
    stop_benchmark()
    predictions = model.predict(X_test_clean)
    scores[name] = get_score(y_test, predictions)
    print(f'{name}: {scores[name]}')

# Split
# Preprocess
# Foreach model type
    # Choose model
    # Measure
# Show graphs