import os
import sys
import time

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from data.model_info import ModelInfo

"""
Used to print one of the trees in a Random Forest model
"""


def main():
	model_path = sys.argv[1]
	model_info = ModelInfo.load(model_path)
	model: RandomForestClassifier = model_info.model

	time_start = time.time()

	plt.figure(figsize=(60, 20), dpi=600)
	tree.plot_tree(model.estimators_[0], filled=True)
	plt.xlim([-0.5, 10])  # adjust x-axis limit
	plt.ylim([-0.5, 4])  # adjust y-axis limit
	plt.rc('font', size=12)  # adjust font size
	plt.rc('xtick', labelsize=12)  # adjust x-axis tick label font size
	plt.rc('ytick', labelsize=12)  # adjust y-axis tick label font size
	plt.rcParams.update({'axes.labelsize': 'large'})  # adjust axis label font size
	plt.rcParams.update({'axes.titlesize': 'large'})  # adjust plot title font size
	plt.rcParams.update({'legend.fontsize': 'large'})  # adjust legend font size
	plt.rcParams.update({'lines.markersize': 4})  # reduce marker size
	plt.savefig(os.path.join(model_path + "/tree.png"), dpi=600, bbox_inches='tight')

	time_end = time.time()

	print("Time taken to print RF tree: " + str(time_end - time_start) + " seconds.")


if __name__ == "__main__":
	main()
