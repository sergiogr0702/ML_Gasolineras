from enum import Enum
from typing import Dict


class ModelType(Enum):
    """
	Lists the different types of models that can be used for prediction
	"""
    SVM = 1
    RANDOM_FOREST = 2
    EXTREME_BOOSTING_TREES = 3
    KNN = 4

    def get_short_name(self) -> str:
        names = self._short_names()
        if self in names:
            return names[self]
        else:
            raise ValueError("Model type " + self.name + " doesn't have a short name")

    @classmethod
    def from_str(cls, string: str):
        string = string.upper()
        for model_type, name in cls._short_names().items():
            if name.upper() == string:
                return model_type

        raise ValueError("Unknown model type " + string)

    @classmethod
    def _short_names(cls) -> Dict:
        """
		Returns a dict that matches each model type with its short name.
		"""
        return {
            cls.SVM: "SVM",
            cls.RANDOM_FOREST: "RF",
            cls.EXTREME_BOOSTING_TREES: "XBT",
            cls.KNN: "KNN"
        }


class Config:
    """
	Class useed to store the config data passed to the program
	"""

    # Path to the input CSV file containing the data
    input_path: str

    # Path to the folder where the output files will be placed
    output_path: str

    # Model used to perform the prediction
    model_type: ModelType

    # Percent of data to use for testing (0-1). If 0, no testing will be performed.
    test_percent: float

    def __init__(self, input_path: str, output_path: str, model: ModelType, test_percent: float):
        self.input_path = input_path
        self.output_path = output_path
        self.model_type = model
        self.test_percent = test_percent
