import os
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from defs.config import Config
from defs.constants import Constants as Cst


class SplitData:
    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any


def create_dataset_for_train(config: Config) -> DataFrame:
    if os.path.isfile(config.input_path):
        dataset = pd.read_csv(config.input_path)
    else:
        raise RuntimeError("Input path is not a file")

    return dataset


def get_split_data(data: DataFrame, test_percent: float) -> SplitData:
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
