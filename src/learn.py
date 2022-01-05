import pandas as pd
from pandas.core.frame import DataFrame

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

def get_sets(dataset, testset, labels, separator=','):
    """Return the train and test sets for the features and labels arrays."""
    if not dataset:
        X, y = load_iris(return_X_y=True, as_frame=True)
    else:
        data = pd.read_csv(dataset, sep=separator, engine='python')
        X, y = split_labels(data, labels)
        # Use separate table as test set if provided
        if testset:
            test_data = pd.read_csv(dataset, sep=separator, engine='python')
            X_test, y_test = split_labels(test_data, labels)
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


def get_score(labels, predictions) -> int:
    """Return the accuracy score from the test set labels vs the predicted labels."""
    return accuracy_score(labels, predictions)
