import pandas
import numpy

from mlcost import learn

PATH = "data/test/{0}.csv"
TEST_TYPES = ['int64','object','float64','float64']

def test_read():
    trainer = learn.Trainer(PATH.format("full"))
    assert trainer.original_data.shape == (6, 4)
    assert trainer.original_targets.shape == (6,)

    assert trainer.original_data.dtypes.tolist() == TEST_TYPES
    assert trainer.original_targets.dtype == 'object'
    assert trainer.original_targets.nunique() == 3

def test_missing_targets():
    trainer = learn.Trainer(PATH.format("na_target"), null_values='?')
    assert pandas.read_csv(PATH.format("na_target"), engine='python', na_values='?').iloc[: , -1].isnull().sum() == 2

    assert trainer.original_data.shape == (4, 4)
    assert trainer.original_targets.shape == (4,)
    assert trainer.original_targets.nunique() == 3

def test_split_data():
    trainer = learn.Trainer(PATH.format("full"))
    assert trainer.train_data.shape[0] == int(trainer.original_data.shape[0] * (1 - trainer.TEST_SIZE))
    assert trainer.train_data.shape[0] + trainer.test_data.shape[0] == int(trainer.original_data.shape[0])
    assert trainer.train_data.shape[1] == int(trainer.original_data.shape[1])

    assert trainer.train_data.shape[0] == trainer.train_target.shape[0]
    assert trainer.test_data.shape[0] == trainer.test_target.shape[0]

def test_split_data_two_files():
    trainer = learn.Trainer(PATH.format("full"), PATH.format("na_target"), null_values='?')
    assert trainer.train_data.shape == (6, 4)
    assert trainer.test_data.shape == (4, 4)
    assert trainer.train_data.shape[0] == trainer.train_target.shape[0]
    assert trainer.test_data.shape[0] == trainer.test_target.shape[0]

def test_remove_missing_categories_rows():
    trainer = learn.Trainer(PATH.format("full"), PATH.format("na_target"), null_values='?')
    trainer._Trainer__drop_missing_values(trainer.categorical_cols)

    assert trainer.train_data.shape == (5, 4)
    assert trainer.test_data.shape == (3, 4)
    assert trainer.train_data.shape[0] == trainer.train_target.shape[0]
    assert trainer.test_data.shape[0] == trainer.test_target.shape[0]

def test_missing_categorical_data():
    trainer = learn.Trainer(PATH.format("full"), null_values='?')
    trainer.clean_data()
    all_data = numpy.concatenate((trainer.train_data, trainer.test_data), axis=0)
    assert all_data.shape[0] == 5
    assert all([i in range(5) for i in all_data[:,0]])

def test_missing_numerical_data():
    trainer = learn.Trainer(PATH.format("full"), null_values='?')
    trainer.clean_data()
    all_data = numpy.concatenate((trainer.train_data, trainer.test_data), axis=0)
    index_in_all = numpy.where(all_data[:,0] == 2)[0]
    index_in_train = numpy.where(trainer.train_data[:,0] == 2)
    changed_value = all_data[index_in_all[0]][2]
    origin_values = numpy.delete(trainer.train_data[:,2], index_in_train[0]) if len(index_in_train) == 1 else trainer.train_data[:,2]
    assert changed_value == origin_values.mean()

def test_one_hot_encoder():
    trainer = learn.Trainer(PATH.format("full"), null_values='?')
    categories = [pandas.unique(trainer.original_data[col].dropna()).tolist() for col in trainer.original_data[trainer.categorical_cols]]
    assert sorted(categories[0]) == ['A', 'B', 'C']

def test_select_label_column():
    pass
