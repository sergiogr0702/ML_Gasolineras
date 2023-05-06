import os
import pickle

from sklearn.preprocessing import MinMaxScaler

from defs.config import ModelType


# Name of the file containing a dumped model
NAME_MODEL_FILE = "model.pkl"
# Name of the file containing a dumped scaler
NAME_SCALER_FILE = "scaler.pkl"
# Name of the file containing model data
NAME_MODEL_DATA_FILE = "model_data.txt"


class ModelInfo:
	"""
	Class used to represent a trained model and some additional information about it
	"""

	model: object
	model_type: ModelType
	scaler: MinMaxScaler

	def __init__(self, model: object, model_type: ModelType, scaler: MinMaxScaler):
		self.model = model
		self.model_type = model_type
		self.scaler = scaler

	@classmethod
	def load(cls, path_dir: str):
		"""
		Creates an instance of this class based on the data contained in the specified folder.
		The data should be the one created by the save() method.
		"""
		# Load dumped model
		model_file_path = os.path.join(path_dir, NAME_MODEL_FILE)
		model = pickle.load(open(model_file_path, "rb"))

		# Load dumped scaler
		scaler_file_path = os.path.join(path_dir, NAME_SCALER_FILE)
		scaler = pickle.load(open(scaler_file_path, "rb"))

		# Load model data
		model_data_file_path = os.path.join(path_dir, NAME_MODEL_DATA_FILE)
		with open(model_data_file_path) as f:
			data = f.readline().split(",")
		model_type = ModelType[data[0]]

		return cls(model, model_type, scaler)

	def save(self, path_dir: str):
		"""
		Saves the model and the additional information to a folder.
		path_dir: Path to the directory where the data will be saved
		"""
		path_file = os.path.join(path_dir, NAME_MODEL_FILE)
		os.makedirs(path_dir, exist_ok=True)
		pickle.dump(self.model, open(path_file, 'wb'))

		# Save the scaler used to scale the data. This is necessary since the data that will be passed to the model
		# in the future must be scaled to the same range.
		path_file = os.path.join(path_dir, NAME_SCALER_FILE)
		pickle.dump(self.scaler, open(path_file, 'wb'))

		with open(os.path.join(path_dir, NAME_MODEL_DATA_FILE), "w") as f:
			f.write(self.model_type.name)
