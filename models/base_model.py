import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from data.model_info import ModelInfo
from defs.config import Config, ModelType
from data.dataset_operations import SplitData
from defs.exceptions import IllegalOperationError
from defs.constants import Constants as Cst
import data.dataset_operations as data_op
from defs.test_metrics import TestMetrics


class BaseModel(ABC):
    """
	Abstract base class that represents a model
	"""

    split_data: SplitData = None
    unscaled_x_test = None

    def train(self, data: DataFrame, config: Config):
        self.split_data = data_op.get_split_data(data, config.test_percent)

        num_cols = self.split_data.x_train.shape[1]
        scaler = MinMaxScaler()
        scaler.fit(self.split_data.x_train.reshape(-1, 1))
        self.split_data.x_train = scaler.transform(self.split_data.x_train.reshape(-1, 1)).reshape(-1, num_cols)
        if self.split_data.x_test is not None:
            self.unscaled_x_test = self.split_data.x_test
            self.split_data.x_test = scaler.transform(self.split_data.x_test.reshape(-1, 1)).reshape(-1, num_cols)

        time_start = time.time()
        self._train_model(self.split_data.x_train, self.split_data.y_train)
        time_end = time.time()

        print("Time taken to train " + self.get_model_type().get_short_name() + ": "
              + str(time_end - time_start) + " seconds.")

        model_info = ModelInfo(self._get_data_to_save(), config.model_type, scaler)
        model_info.save(config.output_path)

    def test(self, config: Config):
        if self.split_data is not None:
            if config.test_percent > 0:
                short_name = self.get_model_type().get_short_name()
                time_start = time.time()
                prediction = self.get_prediction(self.split_data.x_test)
                time_end = time.time()
                print("Time taken to test " + short_name + ": " + str(time_end - time_start) + " seconds.\n")

                labels = [1, 0]
                conf_matrix = confusion_matrix(self.split_data.y_test, prediction, labels=labels)
                plt.figure(figsize=(15, 12))
                sns.set(font_scale=2.5)
                sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 30}, fmt="d",
                            xticklabels=labels, yticklabels=labels)
                plt.xlabel("Prediction")
                plt.ylabel("Rem")
                plt.title("%s" % short_name)
                plt.savefig(os.path.join(config.output_path, "confusion_%s" % short_name + ".png"))
                plt.close()

                # Save test metrics to a file
                test_metrics = TestMetrics.from_testing(conf_matrix, self.split_data.y_test, prediction)
                test_metrics.to_file(os.path.join(config.output_path, Cst.TEST_METRICS_FILE))
            else:
                raise IllegalOperationError("Cannot test model with test_percent = 0")
        else:
            raise IllegalOperationError("The model must be trained before it can be tested")

    @abstractmethod
    def get_model_type(self) -> ModelType:
        ...

    @abstractmethod
    def _get_data_to_save(self) -> object:
        ...

    @abstractmethod
    def _train_model(self, x, y):
        ...

    @abstractmethod
    def get_prediction(self, x):
        ...


    @abstractmethod
    def get_classes(self):
        ...
