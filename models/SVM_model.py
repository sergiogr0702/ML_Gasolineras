from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from models.base_model import BaseModel

from defs.config import ModelType
from defs.exceptions import IllegalOperationError

"""
SVM model
"""


class SVMModel(BaseModel):
    trained_model: "SVC | None"

    def __init__(self, model: object = None):
        self.trained_model = model

    def get_model_type(self) -> ModelType:
        return ModelType.SVM

    def _get_data_to_save(self) -> object:
        return self.trained_model

    def _train_model(self, x, y):
        model = SVC(gamma='auto')

        # Create the parameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'degree': [2, 3, 4],
        }

        print(f"Searching the best hiperparameters combination in SVM with: {param_grid}\n")

        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(x, y)

        self.trained_model = grid_search.best_estimator_


    def get_prediction(self, x):
        if self.trained_model is None:
            raise IllegalOperationError("The model must be trained before predictions can be made.")
        else:
            return self.trained_model.predict(x)

    def get_multi_prediction(self, x):
        if self.trained_model is None:
            raise IllegalOperationError("The model must be trained before predictions can be made.")
        else:
            return self.trained_model.predict_proba(x)

    def get_classes(self):
        if self.trained_model is None:
            raise IllegalOperationError("The model must be trained before classes can be returned.")
        else:
            return self.trained_model.classes_
