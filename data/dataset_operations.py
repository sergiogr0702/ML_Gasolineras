import os
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from defs.config import Config
from defs.constants import Constants as Cst


class SplitData:
    """
	Class that represents data split in 4 groups: Train X, Train Y, Test X and Test Y
	"""

    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any
    # Additional column for clarity when outputting the dataset
    time_test: Any


def create_dataset_for_train(config: Config) -> DataFrame:
    """
	Creates a dataset meant to be used for model traning and testing using the given config
	"""

    if os.path.isfile(config.input_path):
        dataset = pd.read_csv(config.input_path)
    else:
        raise RuntimeError("Input path is not a file")

    return dataset


def get_split_data(data: DataFrame, test_percent: float) -> SplitData:
    """
	Given a DataFrame, returns a version of it split in test/train and x/y data.
	test_percent: Percentage of data that should be used for testing. The rest will be used for training.
	"""

    split_data = SplitData()

    cols_t = [col for col in data.columns if col.startswith(Cst.PREFIX_COLUMN_PRICE)]
    x_all = data[cols_t].values
    le = LabelEncoder()
    y_all = le.fit_transform(data[Cst.NAME_COLUMN_REM].values)

    if test_percent > 0:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_percent, random_state=0)
        for train_indexes, test_indexes in splitter.split(x_all, y_all):
            split_data.x_train, split_data.x_test = x_all[train_indexes].copy(), x_all[test_indexes].copy()
            split_data.y_train, split_data.y_test = y_all[train_indexes].copy(), y_all[test_indexes].copy()

    else:
        split_data.x_train = x_all
        split_data.y_train = y_all
        split_data.x_test = None
        split_data.y_test = None

    return split_data
