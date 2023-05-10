from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from defs.config import ModelType
from models.base_model import BaseModel
from defs.exceptions import IllegalOperationError

"""
K-nearest neighbors model
"""

MAX_SAMPLES_NCA = 1500


class KnnModel(BaseModel):
	trained_model: "KNeighborsClassifier | None"
	trained_nca: "NeighborhoodComponentsAnalysis | None"

	def __init__(self, saved_data=None):
		if saved_data is None:
			self.trained_model = None
			self.trained_nca = None
		else:
			self.trained_model = saved_data[0]
			self.trained_nca = saved_data[1]

	def get_model_type(self) -> ModelType:
		return ModelType.KNN

	def _get_data_to_save(self) -> object:
		return [self.trained_model, self.trained_nca]

	def _train_model(self, x, y):
		x_nca = x
		y_nca = y

		param_grid = {
			'n_neighbors': [3, 5, 7],
			'weights': ['uniform', 'distance'],
			'metric': ['euclidean', 'manhattan'],
			'algorithm': ['ball_tree', 'kd_tree'],
			'n_jobs': [-1]
		}

		print(f"Searching the best hiperparameters combination in KNN with: {param_grid}\n")

		if len(x) > MAX_SAMPLES_NCA:
			splitter = StratifiedShuffleSplit(n_splits=1, test_size=MAX_SAMPLES_NCA, random_state=0)
			for _, indexes in splitter.split(x, y):
				x_nca = x[indexes].copy()
				y_nca = y[indexes].copy()

		nca = NeighborhoodComponentsAnalysis(random_state=0)
		nca.fit(x_nca, y_nca)
		self.trained_nca = nca
		x_transform = self.trained_nca.transform(x)

		knn = KNeighborsClassifier()
		grid_search = GridSearchCV(knn, param_grid, cv=5)
		grid_search.fit(x_transform, y)

		best_params = grid_search.best_params_
		best_model = KNeighborsClassifier(**best_params)
		best_model.fit(x_transform, y)

		self.trained_model = best_model

	def get_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			x_transform = self.trained_nca.transform(x)
			return self.trained_model.predict(x_transform)

	def get_multi_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			x_transform = self.trained_nca.transform(x)
			return self.trained_model.predict_proba(x_transform)

	def get_classes(self):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before classes can be returned.")
		else:
			return self.trained_model.classes_
