"""
File used to store global constants.
Some of them are based on values from the IOT_pi repo.
"""


class Constants:
	# Name of the column in the input CSV containing REM
	NAME_COLUMN_REM = "Rem"
	# Prefix used to create the names of the columns containing power data across different instants of time
	PREFIX_COLUMN_PRICE = "Precio"

	# Name of the output column that contains the attack prediction for single prediction models
	NAME_COLUMN_PREDICTED_REM = "Predicted Rem"

	# Name of the file where model test metrics will be saved
	TEST_METRICS_FILE = "Test results.txt"
