import os
import sys
from typing import List
from pandas import DataFrame
import data.dataset_operations as data_op
from defs.config import Config, ModelType
from defs.constants import Constants as Cst
import main_train_model as main_train_model
from defs.test_metrics import TestMetrics
from defs.utils import get_script_name

"""
Script used to train the models with different parameters in order to figure out which combination is better.
"""


class TrainRun:
    model_type: ModelType

    def __init__(self, model_type: ModelType):
        self.model_type = model_type


def main():
    args = sys.argv

    if len(args) == 4:
        input_path = args[1]
        output_folder = args[2]
        try:
            test_percent = float(args[3]) / 100
        except ValueError:
            print_help()
            return 1

        run(input_path, output_folder, test_percent)
    else:
        print_help()


def print_help():
    print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_folder test_percent\n"
                                                     "input_path: Path to the CSV file containing input data.\n"
                                                     "output_folder: Path to the folder where the trained models and "
                                                     "model stats will be saved.\n"
                                                     "test_percent: Percent of data that should be used to test the "
                                                     "models. Range: (0, 100)")


def _get_output_path(output_folder: str, train_run: TrainRun):
    return os.path.join(output_folder, "run_" + train_run.model_type.get_short_name())


def create_output_csv(output_folder: str, runs: List[TrainRun]):
    columns = ["Model type", "F1 score", "TPc", "TN", "TPi", "FP", "FN"]
    output = DataFrame(columns=columns)

    for train_run in runs:
        metrics = TestMetrics.from_file(os.path.join(_get_output_path(output_folder, train_run),
                                                     Cst.TEST_METRICS_FILE))
        output.loc[len(output)] = [train_run.model_type.name,
                                   metrics.f1_score, metrics.true_positives_correct, metrics.true_negatives,
                                   metrics.true_positives_incorrect, metrics.false_positives,
                                   metrics.false_negatives]

    output.to_csv(os.path.join(output_folder, "multi_train_results.csv"), index=False)


def single_run(train_run: TrainRun, input_path: str, output_folder: str, test_percent: float):
    input_path = input_path
    output_path = _get_output_path(output_folder, train_run)
    model_type = train_run.model_type
    test_percent = test_percent

    config = Config(input_path, output_path, model_type, test_percent)

    print("Training " + train_run.model_type.name)

    dataset = data_op.create_dataset_for_train(config)

    main_train_model.run(config, dataset)


def run(input_path: str, output_folder: str, test_percent: float):
    runs = []

    for mt in [ModelType.SVM, ModelType.RANDOM_FOREST, ModelType.EXTREME_BOOSTING_TREES, ModelType.KNN]:
        runs.append(TrainRun(mt))

    for train_run in runs:
        single_run(train_run, input_path, output_folder, test_percent)

    create_output_csv(output_folder, runs)


if __name__ == "__main__":
    main()
