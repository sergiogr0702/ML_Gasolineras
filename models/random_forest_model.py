import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from defs.config import ModelType
from models.base_model import BaseModel

from defs.exceptions import IllegalOperationError

"""
Random Forest model
"""


class RandomForestModel(BaseModel):
    trained_model: "RandomForestClassifier | None"

    def __init__(self, model: object = None):
        self.trained_model = model

    def get_model_type(self) -> ModelType:
        return ModelType.RANDOM_FOREST

    def _get_data_to_save(self) -> object:
        return self.trained_model

    def _train_model(self, x, y):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        print(f"Searching the best hiperparameters combination in RF with: {param_grid}\n")

        rf = RandomForestClassifier(random_state=1)

        grid_search = GridSearchCV(rf, param_grid=param_grid, n_jobs=-1)
        grid_search.fit(x, y)

        self.trained_model = grid_search.best_estimator_

    def get_prediction(self, x):
        if self.trained_model is None:
            raise IllegalOperationError("The model must be trained before predictions can be made.")
        else:
            return self.trained_model.predict(x)

    def get_classes(self):
        if self.trained_model is None:
            raise IllegalOperationError("The model must be trained before classes can be returned.")
        else:
            return self.trained_model.classes_
