import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

MODEL_TYPES = {
    'Linear': LogisticRegression(max_iter=2000, verbose=1),
    'Forest': RandomForestClassifier(verbose=1),
    'SupportVector': SVC(verbose=1),
    'Neighbors': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'GradientBoost': GradientBoostingClassifier(verbose=1),
    'Neural': MLPClassifier(max_iter=5000, verbose=1),
}

class Trainer():
    DEFAULT_DATASET_NAME = "Iris"
    TEST_SIZE = 0.2
    MAX_CATEGORIES = 10
    RANDOM_STATE = 5


    def __init__(self, data_path, test_path=None, target_label=None, cross_validate=1, separator=',', no_header=False, openml=False, null_values='?'):
        if data_path:
            self.name = data_path.split('/')[-1].split('.')[0].capitalize()
        else:
            self.name = self.DEFAULT_DATASET_NAME

        self.target_label = target_label
        self.read_args = self.__get_read_args(separator, no_header, null_values)
        self.cross_validate_folds = cross_validate

        self.original_data, self.original_targets = self.__read_data(data_path, openml)
        self.__identify_columns()
        self.__drop_missing_values(self.categorical_cols)

        self.__split_test_data(test_path)


    def __get_read_args(self, separator, no_header, null_values):
        args = {"engine": "python"}

        if no_header:
            args["header"] = None

        if separator: 
            args["sep"] = separator

        if null_values: 
            args["na_values"] = null_values
        return args

        
    def __read_data(self, data_path, openml):
        if openml:
            X, y = fetch_openml(data_path, return_X_y=True, as_frame=True)
            self.raw_data = pd.concat((X, y), axis=1, join='inner')
            return X, y
        elif data_path:
            self.raw_data = pd.read_csv(data_path, **self.read_args)
            return self.__split_labels(self.raw_data)
        else:
            self.raw_data = load_iris(return_X_y=False, as_frame=True).data
            return load_iris(return_X_y=True, as_frame=True)


    def __split_test_data(self, test_path):
        """Return the train and test sets for the features and labels arrays."""
        if test_path:
            test_data = pd.read_csv(test_path, **self.read_args)
            self.test_data, self.test_target = self.__split_labels(test_data)
            self.train_data = self.original_data.copy()
            self.train_target = self.original_targets.copy()
        else:
            self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
                self.original_data, self.original_targets,
                test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)

    
    def __split_labels(self, data: DataFrame):
        """Separate the features from the labels.
        
        Parameters
        ----------
        data -- a DataFrame containing features and labels
        """
        if not self.target_label:
            self.target_label = data.columns[-1]

        # Remove rows with missing labels
        data_all_labeled = data.dropna(axis=0, subset=[self.target_label])

        # Separate features and labels
        y = data_all_labeled[self.target_label]
        X = data_all_labeled.drop([self.target_label], axis=1)

        return X, y


    def __drop_missing_values(self, cols):
        """Remove rows with missing values on the given columns."""
        mask = self.original_data[cols].isna().any(axis=1)
        null_indexes_train = self.original_data.index[mask]
        self.original_data.drop(index=null_indexes_train, inplace=True)
        self.original_targets.drop(index=null_indexes_train, inplace=True)


    def __identify_columns(self):
        """Identify numerical and categorical columns that should be used."""
        self.numerical_cols = [col for col in self.original_data if 
                    self.original_data[col].dtype in ['int64', 'float64']]
        self.categorical_cols = [col for col in self.original_data if
                    self.original_data[col].nunique() < self.MAX_CATEGORIES and self.original_data[col].dtype in ["object", "category"]]


    def clean_data(self, log_output=False):
        """Return a clean version of the input data.
        
        Drop categorical features with more than MAX_CATEGORIES and
        translates the others with One-Hot encoding.
        Fill missing numerical values with the mean.

        log_output -- whether to print a summary of the changes to stdout
        """
        original_size = self.train_data.shape, self.test_data.shape

        # Replace empty numerical items with mean and one-shot categorical columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('i', SimpleImputer()), ('s', StandardScaler())]), 
                    self.numerical_cols),
                ('cat', OneHotEncoder(), self.categorical_cols),
            ],
            remainder='drop')

        self.train_data = self.preprocessor.fit_transform(self.train_data)
        self.test_data = self.preprocessor.transform(self.test_data)

        if log_output:
            self.print_summary(original_size)


    def train(self, model):
        model.fit(self.train_data, self.train_target)
        return model.predict(self.test_data)


    def score(self, predictions):
        """Return several classification metrics comparing the test set labels vs the predicted labels."""
        self.report = classification_report(self.test_target, predictions)
        p, r, f, _ = precision_recall_fscore_support(self.test_target, predictions, average='weighted')

        scores = {
            "test_accuracy": balanced_accuracy_score(self.test_target, predictions),
            "test_precision": p,
            "test_f1_score": f,
            "test_recall": r,
            "n_samples": self.original_data.shape[0]
        }

        return scores
    
    def cross_score(self, model):
        fold = StratifiedKFold(n_splits=self.cross_validate_folds, shuffle=True, random_state=self.RANDOM_STATE)
        cv_scores = cross_validate(Pipeline(steps=[('prep', clone(self.preprocessor)), ('model', clone(model))]),
                                self.original_data, self.original_targets,
                                scoring = {
                                    'accuracy': make_scorer(balanced_accuracy_score),
                                    'precision': make_scorer(precision_score, average='weighted'),
                                    'recall': make_scorer(recall_score, average='weighted'),
                                    'f1_score': make_scorer(f1_score, average='weighted')
                                },
                                cv=fold, n_jobs=None)
        cv_scores["n_samples"] = [self.original_data.shape[0]] * len(cv_scores["fit_time"])

        return cv_scores
    
    def print_summary(self, original_size):
        # Print number of features per type before and after
        # Print target categories and number of each in train and test
        # Number of rows before and after

        print("DATA PREPROCESSING SUMMARY")
        print(f"Original data: {self.original_data.memory_usage(index=True, deep=True).sum():.3e} bytes\n")
        
        print(f"\nDiscarded features: {len([col for col in self.original_data if col not in self.numerical_cols + self.categorical_cols])}")
        print(f"Discarded rows for missing labels: {self.raw_data.shape[0] - self.original_data.shape[0]}")
        print(f"Trained numerical features: {self.numerical_cols}")
        print(f"Trained categorical features: {self.categorical_cols}")
        print("\nRemoved rows from missing categorical values - Train:",
            str(original_size[0][0] - self.train_data.shape[0]), ", Test:",
            str(original_size[1][0] - self.test_data.shape[0]))
        print(f"Final train set rows: {self.train_data.shape[0]}, test set rows: {self.test_data.shape[0]}")

        print("\nTarget distribution:")
        counts_train = self.train_target.value_counts(normalize=True, dropna=False)
        counts_test = self.test_target.value_counts(normalize=True, dropna=False)
        print(pd.concat([counts_train.rename('train'), counts_test.rename('test')], axis=1))
        print("---------------------------")
