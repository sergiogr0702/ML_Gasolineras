import sys
from pandas import DataFrame
import data.dataset_operations as data_op
from defs.config import Config, ModelType
from defs.utils import get_script_name
from models.model_factory import ModelFactory

"""
Main script used to create a prediction model given input data containing power use for a certain device. Allows
specifying parameters to determine the type of model to build.
The program will create an output folder containing the model and any other data required by it. It can also
perform testing on the created model, saving stats about the test (such as the confusion matrix).
The model can then be run by calling the main_run_model script.
"""


def main():
    args = sys.argv

    # Parse flags first
    test_percent = 0
    if "-t" in args:
        pos = args.index("-t")
        if pos == len(args) - 1:
            print_help()
            return 1
        try:
            value = int(args[pos + 1])
        except ValueError:
            print_help()
            return 1
        if 0 < value < 100:
            del args[pos:pos + 2]
            test_percent = value / 100
        else:
            print_help()
            return 1

    if len(args) == 4:
        try:
            model = ModelType.from_str(args[3])
        except ValueError:
            print_help()
            return 1

        config = Config(args[1], args[2], model, test_percent)
        run(config)
        return 0
    else:
        print_help()
        return 1


def run(config: Config, dataset: DataFrame = None):
    """
    Trains the model with the specified config.
    dataset: If present, this dataset will be used for training, instead of creating a new one from scratch.
    """
    if dataset is None:
        dataset = data_op.create_dataset_for_train(config)

    # Call the corresponding model training script
    model = ModelFactory().get_model(config.model_type)
    model.train(dataset, config)
    if config.test_percent > 0:
        model.test(config)


def print_help():
    print("Usage: " + get_script_name(
        sys.argv[0]) +  " input_path output_path model group_amount num_groups prediction_type\n"
                        "input_path: Path to the CSV file containing input data, or to a folder containing all the "
                        "data files.\n"
                        "output_path: Folder where the resulting model will be placed\n"
                        "model: Model to train. Possible values are:\n"
                        "  " + ModelType.SVM.get_short_name() + ": Support Vector Machine\n"
                        "  " + ModelType.RANDOM_FOREST.get_short_name() + ": Random forest\n"
                        "  " + ModelType.EXTREME_BOOSTING_TREES.get_short_name() + ": Extreme boosting trees "
                        "  " + ModelType.KNN.get_short_name() + ": K-Nearest Neighbors\n"
                        "Flags:\n"
                        "-t <percent>: Use <percent>% of the data to test the model. Valid values range from 0 to 100 "
                                                                "(both exclusive).")


if __name__ == "__main__":
    main()
