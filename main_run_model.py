import os
import sys
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from data.model_info import ModelInfo
from defs.constants import Constants as Cst
from defs.exceptions import NotEnoughDataError
from defs.utils import get_script_name
from models.model_factory import ModelFactory

"""
Main script used to run an existing model given input data containing power use for a certain device.
It will output a file with its prediction.
"""


def main():
	args = sys.argv

	if len(args) == 4:
		run(args[1], args[2], args[3])
		return 0
	else:
		print_help()
		return 1


def run(input_path: str, output_path: str, model_path: str):
	# Load saved model and its data
	model_info = ModelInfo.load(model_path)
	# Load dataset. The attack column is not expected and will be dropped if it exists.
	dataset = DataFrame.read_csv(input_path)

	if len(dataset) == 0:
		raise NotEnoughDataError("There weren't enough data entries to run the model")

	scaled_dataset = get_scaled_dataset(dataset, model_info.scaler)

	# Create a BaseModel instance to interact with the model
	model = ModelFactory().get_model(model_info.model_type, model_info.model)

	dataset[Cst.NAME_COLUMN_PREDICTED_REM] = model.get_prediction(scaled_dataset)

	# Save prediction to a file
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	dataset.to_csv(output_path, index=False)


def get_scaled_dataset(data: DataFrame, scaler: MinMaxScaler):
	cols_t = [col for col in data.columns if col.startswith(Cst.PREFIX_COLUMN_PRICE)]
	x_data = data[cols_t].values
	num_cols = x_data.shape[1]
	return scaler.transform(x_data.reshape(-1, 1)).reshape(-1, num_cols)

def print_help():
	print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_path model_path\n"
		"input_path: Path to the CSV file containing input data, or to a folder containing all the data files.\n"
		"output_path: Path to the file where the results will be written. It will be created if it doesn't exist, "
		"or overwritten if it does.\n"
		"model_path: Path to the folder containing the model to run, as created by the model training script.\n"
		"Flags:\n")


if __name__ == "__main__":
	main()
