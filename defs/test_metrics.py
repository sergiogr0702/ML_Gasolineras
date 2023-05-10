import numpy as np
from sklearn.metrics import f1_score

"""
Contains the main test metrics obtained after testing a model, as well as methods to read and write it to a file.
"""


class TestMetrics:
    # Attacks labeled as such, with the right attack type
    true_positives_correct: int
    # Attacks labeled as such, but with the wrong type
    true_positives_incorrect: int
    # Normal behavior instances labeled as such
    true_negatives: int
    # Normal behavior instances incorrectly labeled as attacks
    false_positives: int
    # Attack instances incorrectly labeled as normal behavior
    false_negatives: int
    f1_score: float

    def __init__(self, true_positives_correct: int, true_positives_incorrect: int, true_negatives: int,
                 false_positives: int, false_negatives: int, f1_score_val: float):
        self.true_positives_correct = true_positives_correct
        self.true_positives_incorrect = true_positives_incorrect
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.f1_score = f1_score_val

    @classmethod
    def from_testing(cls, conf_matrix, y_test, prediction):
        true_positives_correct = np.sum(np.diag(conf_matrix)[1:])
        true_negatives = conf_matrix[0][0]
        false_positives = np.sum(conf_matrix[0][1:])
        false_negatives = np.sum(row[0] for row in conf_matrix[1:])
        true_positives_incorrect = \
            np.sum(conf_matrix) - true_positives_correct - true_negatives - false_positives - false_negatives
        f1_score_val = f1_score(y_test, prediction, average="micro")

        # noinspection PyTypeChecker
        # (Incorrect assumption of type returned by np.sum())
        return cls(true_positives_correct, true_positives_incorrect, true_negatives, false_positives, false_negatives,
                   f1_score_val)

    @classmethod
    def from_file(cls, path: str):
        with open(path) as f:
            true_positives_correct = int(f.readline().split(": ")[1])
            true_positives_incorrect = int(f.readline().split(": ")[1])
            true_negatives = int(f.readline().split(": ")[1])
            false_positives = int(f.readline().split(": ")[1])
            false_negatives = int(f.readline().split(": ")[1])
            f1_score_val = float(f.readline().split(": ")[1])
            return cls(true_positives_correct, true_positives_incorrect, true_negatives, false_positives,
                       false_negatives, f1_score_val)

    def to_file(self, path: str):
        with open(path, "w") as f:
            f.write("TPc: " + str(self.true_positives_correct) + "\n")
            f.write("TPi: " + str(self.true_positives_incorrect) + "\n")
            f.write("TN: " + str(self.true_negatives) + "\n")
            f.write("FP: " + str(self.false_positives) + "\n")
            f.write("FN: " + str(self.false_negatives) + "\n")
            f.write("F1: " + str(self.f1_score))
