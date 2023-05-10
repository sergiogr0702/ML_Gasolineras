import os
import sys
from defs.constants import Constants as Cst
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from defs.utils import get_script_name

pd.set_option('display.max_columns', None)

"""
Script used to visualize certain parameters of the dataset.
"""


def main():
    args = sys.argv
    matplotlib.rcParams['figure.figsize'] = (9.0, 6.0)

    if len(args) == 3:
        input_path = args[1]
        output_folder = args[2]

        visualize(input_path, output_folder)
    else:
        print_help()


def print_help():
    print("Usage: " + get_script_name(sys.argv[0]) + " input_path output_folder\n"
                                                     "input_path: Path to the CSV file containing input data.\n"
                                                     "output_folder: Path to the folder where the plots and results "
                                                     "will be save.\n")


def visualize(input_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder + "/visualization"):
        os.makedirs(output_folder + "/visualization")

    output_folder = output_folder + "/visualization"

    dataset = pd.read_csv(input_path)

    generate_statidistics(dataset, output_folder)
    generate_categorical_data_importance(dataset, output_folder)
    generate_gas_data_importance(dataset, output_folder)
    generate_dataset_histograms(dataset, output_folder)
    generate_plot_for_precio_g5(dataset, output_folder)


def generate_plot_for_precio_g5(dataset, output_folder):
    fare_bins = np.arange(0, 1.8, 1.8 / 20)
    plt.title('Precio_g5')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Precio_g5, bins=fare_bins, color="b",
                 kde=True, stat="density", multiple="stack", label="dm")
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Precio_g5, bins=fare_bins, color="r",
                 kde=True, stat="density", multiple="stack", label="OM")
    plt.legend(loc='upper right')

    plt.savefig(f"{output_folder}/preciog5_plot.png")
    plt.close()


def generate_dataset_histograms(dataset, output_folder):
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 25))
    fare_bins = np.arange(0, 1.8, 1.8 / 20)

    cols_t = [col for col in dataset.columns if col.startswith(Cst.PREFIX_COLUMN_PRICE)]

    # Plot for Provincia
    ax = axes[0 // 3, 0 % 3]
    ax.set_title('Provincia')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Provincia, color="b",
                 kde=True, stat="density", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Provincia, color="r",
                 kde=True, stat="density", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

    # Plot for Municipio
    ax = axes[1 // 3, 1 % 3]
    ax.set_title('Municipio')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Municipio, color="b",
                 kde=True, stat="density", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Municipio, color="r",
                 kde=True, stat="density", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

    # Plot for Localidad
    ax = axes[2 // 3, 2 % 3]
    ax.set_title('Localidad')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Localidad, color="b",
                 kde=True, stat="density", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Localidad, color="r",
                 kde=True, stat="density", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

    # Plot for Direccion
    ax = axes[3 // 3, 3 % 3]
    ax.set_title('Direccion')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Direccion, color="b",
                 kde=True, stat="density", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Direccion, color="r",
                 kde=True, stat="density", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

    # Plot for Rotulo
    ax = axes[4 // 3, 4 % 3]
    ax.set_title('Rotulo')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Rotulo, color="b",
                 kde=True, stat="density", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Rotulo, color="r",
                 kde=True, stat="density", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

    # Plot for Margen
    ax = axes[5 // 3, 5 % 3]
    ax.set_title('Margen')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Margen, color="b",
                 kde=True, stat="density", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Margen, color="r",
                 kde=True, stat="density", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Plot for Tipo_venta
    ax = axes[6 // 3, 6 % 3]
    ax.set_title('Tipo_venta')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Tipo_venta, color="b",
                 kde=True, stat="count", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Tipo_venta, color="r",
                 kde=True, stat="count", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')

    max_count_dm = dataset[dataset['Rem'] == 'dm'].Tipo_venta.value_counts().max()
    max_count_om = dataset[dataset['Rem'] == 'OM'].Tipo_venta.value_counts().max()

    y_max = max(max_count_dm, max_count_om)
    ax.set_ylim(0, y_max * 1.1)  # Increase the upper limit by 10%

    # Plot for Horario
    ax = axes[7 // 3, 7 % 3]
    ax.set_title('Horario')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Horario, color="b",
                 kde=True, stat="count", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Horario, color="r",
                 kde=True, stat="count", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])

    max_count_dm = dataset[dataset['Rem'] == 'dm'].Horario.value_counts().max()
    max_count_om = dataset[dataset['Rem'] == 'OM'].Horario.value_counts().max()

    y_max = max(max_count_dm, max_count_om)
    ax.set_ylim(0, y_max * 1.1)  # Increase the upper limit by 10%

    i = 8
    for colName in cols_t:
        ax = axes[i // 3, i % 3]
        ax.set_title(colName)
        sns.histplot(dataset[dataset['Rem'] == 'dm'][colName], bins=fare_bins, color="b",
                     kde=True, stat="density", multiple="stack", label="dm", ax=ax)
        sns.histplot(dataset[dataset['Rem'] == 'OM'][colName], bins=fare_bins, color="r",
                     kde=True, stat="density", multiple="stack", label="OM", ax=ax)
        ax.legend(loc='upper right')
        ax.set_xlabel('')
        ax.set_ylabel('')

        i += 1

    # Plot for Rem
    ax = axes[17 // 3, 17 % 3]
    ax.set_title('Rem')
    sns.histplot(dataset[dataset['Rem'] == 'dm'].Rem, color="b",
                 kde=True, stat="count", multiple="stack", label="dm", ax=ax)
    sns.histplot(dataset[dataset['Rem'] == 'OM'].Rem, color="r",
                 kde=True, stat="count", multiple="stack", label="OM", ax=ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f"{output_folder}/dataset_histograms.png")

    plt.close()


def generate_statidistics(dataset, output_folder):
    with open(f"{output_folder}/dataset_stadistics.txt", "w") as f:
        f.write(f"Dataset stadistics\n")
        f.write(f"{dataset.shape}\n")
        f.write(f"{dataset.columns}\n")
        f.write(f"{dataset.dtypes}\n")
        f.write(f"{dataset.isnull().sum() / len(dataset) * 100}\n")

        f.write(f"{dataset.describe()}\n")


def generate_categorical_data_importance(dataset, output_folder):
    with open(f"{output_folder}/categorical_data_importance.txt", "w") as f:

        f.write(f"\nCategorical data\n")

        le = LabelEncoder()
        df = dataset.copy()

        df = df.loc[:, ['Codigo_postal', 'Tipo_venta', 'Rem', 'Localidad', 'Municipio', 'Provincia', 'Rotulo',
                        'Horario', 'Precio_g1', 'Precio_g2', 'Precio_g3', 'Precio_g4', 'Precio_g5', 'Precio_g6',
                        'Precio_g7', 'Precio_g8', 'Precio_g9']]

        for col in ['Codigo_postal', 'Tipo_venta', 'Rem', 'Localidad', 'Municipio', 'Provincia', 'Rotulo', 'Horario']:
            df[col] = le.fit_transform(df[col])

        X = df.drop(columns=['Rem'])

        df = dataset.loc[:, ['Rem']]
        y = le.fit_transform(df['Rem'].values)

        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        clf.fit(X, y)

        importances = clf.feature_importances_

        feature_names_categorical = ['Codigo_postal', 'Tipo_venta', 'Localidad', 'Municipio', 'Provincia', 'Rotulo',
                                     'Horario']

        cont = 0
        for i, imp in enumerate(importances):
            if i < len(feature_names_categorical):
                cont += imp
                f.write(f"Attribute {feature_names_categorical[i]}: {imp}\n")

        f.write(f"\nImportance of all categorical data {cont}\n")


def generate_gas_data_importance(dataset, output_folder):
    with open(f"{output_folder}/numerical_data_importance.txt", "w") as f:
        f.write(f"Numerical data\n")

        le = LabelEncoder()
        df = dataset.copy()

        df = df.loc[:, ['Rem', 'Precio_g1', 'Precio_g2', 'Precio_g3', 'Precio_g4', 'Precio_g5', 'Precio_g6',
                        'Precio_g7', 'Precio_g8', 'Precio_g9']]

        X = df.drop(columns=['Rem'])

        df = dataset.loc[:, ['Rem']]
        y = le.fit_transform(df['Rem'].values)

        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        clf.fit(X, y)

        importances = clf.feature_importances_

        features = ['Precio_g1', 'Precio_g2', 'Precio_g3', 'Precio_g4', 'Precio_g5', 'Precio_g6', 'Precio_g7',
                    'Precio_g8', 'Precio_g9']

        cont = 0
        for i, imp in enumerate(importances):
            cont += imp
            f.write(f"Attribute {features[i]}: {imp}\n")

        f.write(f"\nImportance of all prices data {cont}\n")


if __name__ == "__main__":
    main()
