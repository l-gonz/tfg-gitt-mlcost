import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

MODEL_TYPES = {
    'Linear': LogisticRegression(),
    'Forest': RandomForestClassifier(),
    'SupportVector': SVC(),
    'Neighbors': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'GradientBoost': GradientBoostingClassifier(),
    'Neural': MLPClassifier(),
}

class Trainer():
    DEFAULT_DATASET_NAME = "Iris"
    TEST_SIZE = 0.2
    MAX_CATEGORIES = 10


    def __init__(self, data_path, test_path=None, target_label=None, separator=',', no_header=False, null_values=None):
        if data_path:
            self.name = data_path.split('/')[-1].split('.')[0].capitalize()
        else:
            self.name = self.DEFAULT_DATASET_NAME

        self.data_path = data_path
        self.test_path = test_path
        self.target_label = target_label

        self.read_args = {"engine": "python"}
        # Make headers if data had none
        if no_header:
            self.read_args["header"] = None
            with open(self.data_path) as f:
                self.read_args["names"] = ["Col" + str(i) for i in range(f.readline().count(separator))] + ["Target"]
        if separator: self.read_args["sep"] = separator
        if null_values: self.read_args["na_values"] = null_values

        self.original_data, self.original_targets = self.__read_data()
        self.__split_test_data()
        self.__identify_columns()

        
    def __read_data(self):
        if self.data_path:
            self.raw_data = pd.read_csv(self.data_path, **self.read_args)
            return self.__split_labels(self.raw_data)
        else:
            self.raw_data = load_iris(return_X_y=False, as_frame=True).data
            return load_iris(return_X_y=True, as_frame=True)


    def __split_test_data(self):
        """Return the train and test sets for the features and labels arrays."""
        if self.test_path:
            test_data = pd.read_csv(self.test_path, **self.read_args)
            self.test_data, self.test_target = self.__split_labels(test_data)
            self.train_data = self.original_data.copy()
            self.train_target = self.original_targets.copy()
        else:
            self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
                self.original_data, self.original_targets,
                train_size=1-self.TEST_SIZE, test_size=self.TEST_SIZE)

    
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
        mask = self.train_data[cols].isna().any(axis=1)
        null_indexes_train = self.train_data.index[mask]
        self.train_data.drop(index=null_indexes_train, inplace=True)
        self.train_target.drop(index=null_indexes_train, inplace=True)
        
        mask = self.test_data[cols].isna().any(axis=1)
        null_indexes_test = self.test_data.index[mask]
        self.test_data.drop(index=null_indexes_test, inplace=True)
        self.test_target.drop(index=null_indexes_test, inplace=True)


    def __identify_columns(self):
        """Identify numerical and categorical columns that should be used."""
        self.numerical_cols = [col for col in self.train_data if 
                    self.train_data[col].dtype in ['int64', 'float64']]
        self.categorical_cols = [col for col in self.train_data if
                    self.train_data[col].nunique() < self.MAX_CATEGORIES and self.train_data[col].dtype == "object"]


    def clean_data(self, log_output=False):
        """Return a clean version of the input data.
        
        Drop categorical features with more than MAX_CATEGORIES and
        translates the others with One-Hot encoding.
        Fill missing numerical values with the mean.

        log_output -- whether to print a summary of the changes to stdout
        """
        dropped_cols = [col for col in self.train_data if col not in self.numerical_cols + self.categorical_cols]
        original_row_count = self.train_data.shape[0], self.test_data.shape[0]

        # Remove categorical columns with too many categories
        self.train_data.drop(columns=dropped_cols, inplace=True)
        self.test_data.drop(columns=dropped_cols, inplace=True)
        self.__drop_missing_values(self.categorical_cols)

        # Replace empty numerical items with mean and one-shot categorical columns
        numerical_transformer = SimpleImputer()
        categorical_transformer = OneHotEncoder(categories=[pd.unique(self.original_data[col].dropna()).tolist() for col in self.original_data[self.categorical_cols]])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])

        self.train_data = preprocessor.fit_transform(self.train_data)
        self.test_data = preprocessor.transform(self.test_data)

        if log_output:
            self.print_summary(dropped_cols, original_row_count)


    def train(self, model):
        model.fit(self.train_data, self.train_target)
        return model.predict(self.test_data)


    def score(self, predictions) -> float:
        """Return the accuracy score from the test set labels vs the predicted labels."""
        self.report = classification_report(self.test_target, predictions)
        p, r, f, _ = precision_recall_fscore_support(self.test_target, predictions, average='weighted')
        return {
            "accuracy": accuracy_score(self.test_target, predictions),
            "precision": p,
            "f-beta": f,
            "recall": r,
        }

    
    def print_summary(self, dropped_cols, original_size):
        # Print number of features per type before and after
        # Print target categories and number of each in train and test
        # Number of rows before and after

        print("DATA PREPROCESSING SUMMARY")
        # print("Original data:\n")
        # print(self.original_data.describe())
        print(f"\nDiscarded features: {dropped_cols}")
        print(f"Discarded rows for missing labels: {self.raw_data.shape[0] - self.original_data.shape[0]}")
        print(f"Trained numerical features: {self.numerical_cols}")
        print(f"Trained categorical features: {self.categorical_cols}")
        print("\nRemoved rows from missing categorical values - Train:",
            str(original_size[0] - self.train_data.shape[0]), ", Test:",
            str(original_size[1] - self.test_data.shape[0]))
        print(f"Final train set rows: {self.train_data.shape[0]}, test set rows: {self.test_data.shape[0]}")

        print("\nTarget distribution:")
        counts_train = self.train_target.value_counts(normalize=True, dropna=False)
        counts_test = self.test_target.value_counts(normalize=True, dropna=False)
        print(pd.concat([counts_train.rename('train'), counts_test.rename('test')], axis=1))
        print("---------------------------")
