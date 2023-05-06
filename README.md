# Gasolineras

Este proyecto contiene el código para etiquetar, limpiar, guardar datos del dataset de las gasolineras usado en 
el entrenamiento y testeo de modelos para clasificar las gasolineras en función del tipo de servicio (Rem).
Mira también el proyecto [Proyecto_Gasolineras](https://github.com/sergiogr0702/ProyectoGasolineras) para ver el código del programa principal.

# Estructura de carpetas

- data: Scripts usados para procesar y modificar datos
- defs: Definiciones comunes usadas en todo el proyecto
- models: Scripts usados para entrenar y testear modelos

# Hyperparámetros

Este proyecto contiene múltiples constantes y parámetros que afectan a la forma en la que se ejecutan los modelos o se construyen los datasets. Se listan aquí para conveniencia. Para una explicación de lo que hacen, mira los comentarios de los archivos correspondientes o la documentación de los métodos de la librería.

- Data
	- dataset_operations
		- `PERCENT_ATTACK_CHECK`
		- `PERCENT_ATTACK_THRESHOLD`
- Models
	- random_forest_model
		- `RandomForestClassifier` instanciación de parámetros
	- extreme_boosting_trees_model
		- `xgb` parámetros de entrenamiento
	- knn_model
		- `MAX_SAMPLES_NCA`
		- `KNeighborsClassifier` instanciación de parámetros

No obstante, estos parámetros no deberían ser modificados ya que por defecto los algoritmos están configurados para obtener los mejores resultados posibles. Si se desea modificar alguno, se recomienda hacerlo en un nuevo archivo y no modificar los valores por defecto.

# Instalación

En primer lugar, es necesario clonar el proyecto en el directorio deseado. Para ello, ejecuta el siguiente comando en la terminal:

```
git clone https://github.com/sergiogr0702/ML_Gasolineras.git
```

A continuación, es recomendable crear un entorno virtual para instalar las dependencias del proyecto. Para ello, ejecuta el siguiente comando en la terminal:

```
python -m venv venv
```

Una vez creado el entorno virtual, es necesario activarlo. Para ello, ejecuta el siguiente comando en la terminal:

```
venv\Scripts\activate.bat
```

Por último, es necesario instalar las dependencias del proyecto. Para ello, ejecuta el siguiente comando en la terminal:

```
venv\Scripts\pip install -r requirements.txt
```

# Ejecución

Existen diferentes scripts que se pueden ejecutar para realizar diferentes tareas.

- `main_visualize_dataset.py`: Muestra distintos gráfico con la distribución de los datos del dataset además de estadísticas sobre los mismos.
- `main_train_model-py`: Entrena un modelo determinado con los datos del dataset y guarda los resultados del testeo y el modelo entrenado.
- `main_train_multiple_models.py`: Entrena todos los modelos disponibles con los datos del dataset y guarda los resultados del testeo y los modelos entrenados.
- `main_run_model.py`: Ejecuta un modelo determinado con los datos del dataset y guarda los resultados de la ejecución.
- `main_print_tree.py`: Muestra el árbol de decisión generado a partir del modelo Random Forest.

Para ejecutar un script, ejecuta el siguiente comando en la terminal mientras en entorno virtual está activado:

```
venv\Scripts\python <nombre_script>.py
```