from defs.config import ModelType
from models.SVM_model import SVMModel
from models.extreme_boosting_trees_model import ExtremeBoostingTreesModel
from models.knn_model import KnnModel
from models.random_forest_model import RandomForestModel


class ModelFactory:
	"""
	Allows creating model instances given a ModelType value
	"""

	def __init__(self):
		pass

	def get_model(self, model_type: ModelType, trained_model: object = None):
		"""
		Returns a new model of the given type.
		trained_model: If specified, the model object returned will be trained using the specified specific model.
		The type of this model object must match the expected type (eg. SVM model if model_tpye is ModelType.SVM).
		"""

		if model_type == ModelType.SVM:
			return SVMModel(trained_model)
		elif model_type == ModelType.RANDOM_FOREST:
			return RandomForestModel(trained_model)
		elif model_type == ModelType.EXTREME_BOOSTING_TREES:
			return ExtremeBoostingTreesModel(trained_model)
		elif model_type == ModelType.KNN:
			return KnnModel(trained_model)
		else:
			raise NotImplementedError("Model type " + str(model_type) + " has not been implemented.")
