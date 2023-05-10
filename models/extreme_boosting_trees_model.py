import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
# noinspection PyUnresolvedReferences
# Necessary for type checks
from xgboost import Booster

from defs.config import ModelType
from models.base_model import BaseModel

from defs.exceptions import IllegalOperationError

"""
Extreme Boosting Trees model
"""

TRAIN_ROUNDS = 70


class ExtremeBoostingTreesModel(BaseModel):
    trained_model: "Booster | None"

    def __init__(self, model: object = None):
        self.trained_model = model

    def get_model_type(self) -> ModelType:
        return ModelType.EXTREME_BOOSTING_TREES

    def _get_data_to_save(self) -> object:
        return self.trained_model

    def _train_model(self, x, y):
        x_train_matrix = xgb.DMatrix(self.split_data.x_train, label=self.split_data.y_train)
        num_labels = max(self.split_data.y_test) + 1

        # Define the parameter grid
        param_grid = {
            'booster': ['gbtree'],
            'eta': [0.3, 0.1, 0.01],
            'objective': ['multi:softmax'],
            'max_depth': [5, 7, 10],
            'num_class': [num_labels]
        }

        print(f"Searching the best hiperparameters combination in XBT with: {param_grid}\n")

        best_score = float('-inf')
        best_params = None

        # Perform manual grid search
        for params in ParameterGrid(param_grid):
            model = xgb.train(params, x_train_matrix, TRAIN_ROUNDS)
            x_test_matrix = xgb.DMatrix(x)
            y_pred = model.predict(x_test_matrix)
            score = accuracy_score(y, y_pred)

            if score > best_score:
                best_score = score
                best_params = params

        # Train the final model with the best parameters
        final_model = xgb.train(best_params, x_train_matrix, TRAIN_ROUNDS)
        self.trained_model = final_model

    def get_prediction(self, x):
        if self.trained_model is None:
            raise IllegalOperationError("The model must be trained before predictions can be made.")
        else:
            x_test_matrix = xgb.DMatrix(self.split_data.x_test, label=self.split_data.y_test)
            y_predict_log = self.trained_model.predict(x_test_matrix)

            return np.round(y_predict_log).astype(int)

    def get_multi_prediction(self, x):
        raise IllegalOperationError("This model doesn't support multiple predictions.")

    def get_classes(self):
        raise IllegalOperationError("This model doesn't support getting its class list.")
