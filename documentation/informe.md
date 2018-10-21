# Predicciones sobre de cancer de mama

## Descripción del problema

A partir de un dataset de muestras de tejidos que pueden ser benignos o malignos se realizó dos modelos de Machine Learning en lenguaje de programación _Python 3_, Random Forest y Redes Neuronales, para intentar predecir con el menor margen de error posible si dados ciertos datos, se puede deducir que el tejido es maligno o benigno. El dataset utilizado para el entrenamiento de estos modelos puede ser encontrado [aquí](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

## Descripción de los modelos

### Red Neuronal

Las redes neuronales son un tipo de modelo del área denominada "Machine Learning". Su popularidad se ha visto incrementada en los últimos años debido a su versatilidad para tratar problemas tanto de clasificación como regresión, además, los recursos computacionales contemporáneos nos permiten explotar esta tecnología. Una red neuronal se inspira en la hipótesis que la unidad básica de procesamiento mental es la neurona.

Las redes neuronales estan descritas especialmente por dos características: su topología y el modo de activación de las neuronas. En cuanto a topología, se puede pensar en una red neuronal como un grafo en capas con aristas que transportan información. Las redes neuronales poseen mínimo una capa de entrada de datos y otra capa de salida que representan las predicciones del modelo. 

Las funciones de activación corresponden a criterios de cada neurona en donde dependiendo de las entradas que lleguen a ella, se activa o no. La manera en que las neuronas se activan y los datos que propagan en sus salidas dependen de la función de activación. Algunas funciones de activación famosas corresponden a ReLU, Softplus, Softmax, etc.

Para la creación y entrenamiento de la red neuronal en este proyecto, se utilizo la biblioteca tensorflow creada por Google la cual nos brinda gran cantidad de opciones para creación de modelos de learning.  También se hizo uso de la biblioteca Keras como interfaz intermediaria.  Tensorflow recibe los siguientes parámetros:

Numero de capas:  Cantidad de capas que el modelo va a tener.

Neuronas por capa: Cantidad de neuronas de cada capa intermedia.

Neuronas de salida: Cantidad de neuronas en la capa de salida.

Función de activación: Función de activación de las neuronas intermedias.

Función de activación de salida: Función de activación de las neuronas de salida.

### Decission Tree

Un arbol de decisión es una función que toma un vector o dataset como parámetro de entrada y genera un resultado llamado decisión, esta decisión puede ser un valor discreto o continuo.

#### Nodos del árbol

Existen dos tipos de nodos en un árbol de decisión, ramas y hojas, las ramas contienen los predicados o "preguntas" que permiten tomar una decisión al algoritmo, por otra parte, las hojas contienen la decisión a la que se llega mediante los predicados en las ramas del árbol.

#### Selección de predicados

La selección de un predicado en un nivel del árbol determina que tanto se disminuirá la incertidumbre de los datos en el siguiente nivel para así poder llegar una decisión lo más acertada posible, es por esto que elegir el mejor predicado para dividir el árbol es la tarea más importante. Para la elección de cada predicado, se calcula la Entropía del sistema y la ganancia de cada atributo.

#### Entropia de Shannon

La entropia de un sistema se refiere a la medida de incertidumbre. Un sistema totalmente parcial no posee incertidumbre probabilistica, por lo cual su entropía es cero. Mientras tanto, un sistema con igual probabilidad en cada uno de sus posibilidades es un sistema inparcial cuya entropía corresponde a uno. La fórmula usada para calcular la entropía de la variable V  con d cantidad distinto de valores es la siguiente:

#### Ganancia

La ganancia se refiere a la cantidad de entropia resultante de un sistema luego de que el conjunto de datos haya sido dividido a base de un atributo en específico. Su fórmula esta dada por la entropía menos el residuo que otorga la selección del atributo. El residuo es calculado de la siguiente manera:

El diseño e implementación del algoritmo Decission Tree esta basado en el algoritmo de la página 702 del libro "**_Artifical Intelligence a Modern Aproach Third Edition_ (AIMA)**" 

### Random Forest

Random Forest es un modelo de learning que utiliza como base los árboles de decisión para emitir una predicción. Su mecanismo consiste en crear __n__ cantidad de árboles de decisión y realizar una votación sobre las predicciones de todos, por mayoría se escoge la predicción a devolver.  El subconjunto de datos para entrenar cada árbol se escoge de manera aleatoria con reemplazo de entre todos los datos.  Para la implementación del random forest de este proyecto, el tamaño del subconjunto de datos es equivalente al del conjunto total y la complejidad corresponde al número de árboles.

La estrategia para escoger la mejor partición de cada nodo cambia con respecto al algoritmo de _Tree Bagging_. En esta implementación no se toma en cuenta todos los atributos (columnas) que tenga el dataset del árbol, sino se escoge de manera aleatoria un subconjunto de atributos y entre ellos se hace el análisis de ganancia. El tamaño de este subconjunto está dado por la raíz cuadrada redondeada al menor de la cantidad total de atributos del dataset. Para mayor información al respecto, el estándar del algoritmo usado en random forest lo puede encontrar [aquí](https://en.wikipedia.org/wiki/Random_forest).



## Guía de instalación y ejecución

El proyecto esta construido en el lenguaje de programación Python. Recomendamos el uso de Python 3.5 debido a que algunas bibliotecas que utiliza el proyecto, aún no se encuentran disponibles en versiones posteriores. Para obtener el proyecto, proceda a descargarlo de https://github.com/JuViquez/Proyecto1-IA-Cancer-Predictions.git o bien realice un _git clone_. Para instalar las dependencias se utiliza pip. A continuación, la lista de comandos a ejecutar en el orden correspondiente para ejecutar el proyecto:

```python
pip install pandas
pip install numpy
pip install keras
pip install -U scikit-learn
pip install pytest
pip install pytest-cov
```

Ejemplos de ejecución de código:

```python
#Ejemplo de árbol:
python trainer.py --prefijo breast_cancer.csv --indice_columna_y 1 --porcentaje-pruebas 0.15 arbol --umbral-poda 0.2
#Ejemplo de red:
python Trainer.py --prefijo breast_cancer.csv --indice_columna_y 1 --porcentaje-pruebas 0.15 red-neuronal --numero-capas 5 --unidades-por-capa 20 --funcion-activacion relu --funcion-activacion-salida softmax --iteraciones-optimizador 20

```

## Análisis de resultados

### Red Neuronal

Parte del trabajo de optimización de modelos basados en redes neuronales es encontrar el equilibrio perfecto en la topología y función de activación, de tal forma que el modelo sea lo suficientemente eficiente. El equipo de trabajo optó por trabajar con ReLU como función de activación de las neuronas en capas intermedias. Para la activación de neuronas de salida, se optó por emplear softmax debido a que su comportamiento de probabilidad nos facilita la interpretación en modelos de clasificación.

Los hiperparámetros a configurar cada vez que se crea un modelo son número de capas y neuronas por capa. El equipo de trabajo realizó un análisis de resultados con distintos valores para estos dos parámetros. Para el número de capas se definió un conjunto de 4, 5 y 6. Para las neuronas por capa, inicialmente se prueba cada modelo con 15 neuronas por capa, incrementando este número de 5 en 5 hasta llegar a un tope de 30. La metodología de las pruebas consiste en ejecutar tres veces cada iteración y promediar el error de entrenamiento y de validación. Los resultados en detalles pueden ser consultados en el archivo **analisis_resultados.xlsx** en la pestaña Redes - Capas y Unidades.

| Capas Unidades | Promedio error Entrenamiento | Promedio error Validación |
| -------------- | ---------------------------- | ------------------------- |
| 4 - 15         | 0.0132                       | 0.0352                    |
| 4 - 20         | 0.0104                       | 0.0156                    |
| 4 - 25         | 0.0058                       | 0.0431                    |
| 4 - 30         | 0.0080                       | 0.0235                    |
| 5 - 15         | 0.0094                       | 0.03133                   |
| 5 - 20         | 0.0076                       | 0.0235                    |
| 5 - 25         | 0.00372                      | 0.0470                    |
| 5 - 30         | 0.0041                       | 0.0195                    |
| 6 - 15         | 0.0089                       | 0.0252                    |
| 6 - 20         | 0.0035                       | 0.0313                    |

![](https://github.com/JuViquez/Proyecto1-IA-Cancer-Predictions/blob/master/documentation/charts/chart_redes.PNG?raw=true)

Las iteraciones de 6 capas con 25 y 30 unidades por capa no se realizaron debido a que el costo de la complejidad de la red neuronal era muy grande para la ganancia mínima, con respecto a los modelos anteriores, que se iba a obtener de estas configuraciones. Dado estos resultados, el equipo de trabajo llegó a la conclusión que la mejor configuración para crear el modelo corresponde a 4 capas con 20 unidades, esto debido a que el error de entrenamiento y de validación observado son lo suficientemente pequeños para aceptarlos y aunque el modelo de 5 capas con 30 unidades posea mejor registro en error de entrenamiento, su nivel de complejidad debilita esta elección en comparación a la de 4 capas con 20 unidades.

### Random Forest

#### Elección del subconjunto de pruebas

El modelo de Random Forest recibe como parámetro el porcentaje del dataset original que será usado para validar el modelo en la última fase del programa. Para ello se hizo un análisis del rendimiento del modelo con distintos porcentajes de partición. Se tomaron en cuenta únicamente 10%, 15%, 20%, 25% y 30% para realizar estas pruebas, ya que el estándar de learning oscila entre estos rangos.  La medida de rendimiento corresponde a la media del error de entrenamiento, el error de validación y la complejidad del modelo presente en siete iteraciones distintas por cada porcentaje. Los resultados completos se pueden observar en el archivo **analisis_resultados.xlsx** en la pestaña Arbol - Porcentaje de Pruebas.

| Porcentajes | Promedio de error entrenamiento | Promedio de error validación | Promedio de  complejidad |
|:-----------:|:-------------------------------:|:----------------------------:|:------------------------:|
| 10%         | 0.0072                          | 0.0751                       | 4                        |
| 15%         | 0.0065                          | 0.0554                       | 4.42                     |
| 20%         | 0.0072                          | 0.0687                       | 4.57                     |
| 25%         | 0.0117                          | 0.0724                       | 3                        |
| 30%         | 0.0223                          | 0.0734                       | 3.71                     |

![](https://github.com/JuViquez/Proyecto1-IA-Cancer-Predictions/blob/master/documentation/charts/chart_err_val.PNG?raw=true)
![](https://github.com/JuViquez/Proyecto1-IA-Cancer-Predictions/blob/master/documentation/charts/chart_err_ent.PNG?raw=true)
Los resultados son concisos y contundentes, el porcentaje de partición que asegura la menor tasa de error tanto en entrenamiento como en validación es 15% y a pesar de que la media de su complejidad es la segunda más elevada, estamos tratando con complejidades de alrededor de 4.5, lo cual sigue siendo manejable en términos computacionales.

#### Criterio de poda

Una característica particular de los árboles de decisión es la capacidad de poder podarlos. Podar un árbol de decisión es convertir en hoja un node cuyas ramas son sólo hojas y la ganancia de información en esa partición no es considerable, por lo cual se toma en cuenta la pluralidad de las hojas como la predicción. El dilema se encuentra en definir el umbral por el cual decidimos que podar el árbol es una buena decisión. Para ello el equipo de trabajo realizó pruebas con distintos umbrales de poda. Por cada umbral se ejecutan cinco modelos a los cuales se les calcula su error de entrenamiento y validación antes y después de ser podados. Los resultados completos se pueden observar en el archivo **analisis_resultados.xlsx** en la pestaña Arbol - Poda.

| Criterio Poda | Promedio diferencia de entrenamiento | Promedio de diferencia Validación |
|:-------------:|:------------------------------------:|:---------------------------------:|
| 0.2           | 0.00145                              | 0                                 |
| 0.3           | 0.0057                               | 0                                 |
| 0.4           | 0.0032                               | 0                                 |

Tal y como se puede observar en los resultados, podar el árbol no posee ganancia alguna en el modelo. No tomamos en cuenta criterios menores a 0.2 ya que nos parecen insignificantes y pocos nodos realmente llegan a tener ese nivel de ganancia. Por otra parte, podar el árbol con niveles mayores a 0.4 incrementa drásticamente la diferencia de error de entrenamiento, mientras que la diferencia de error de validación permanece casi nula. Podar árboles es una estrategia empleada para combatir árboles complejos que tienden a _overfitting_, pero los random forest son un modelo bastante bueno evitando estos casos, por lo tanto la ganancia no se ve reflejada.

### Covertura de pruebas



### Distribución de trabajo

El trabajo de ambos integrantes fue equivalente, por lo tanto se recomienda repartir la nota en partes iguales.

**Julio Viquez Murillo 2015013680:**

- Implementación y pruebas redes neuronales

- Implementación y pruebas de cross validation

- Implementación y pruebas de Trainer.py / Program.py

- Diseño del programa

**José Antonio Salas Bonilla 2015013633:**

- Implementación y pruebas de árboles de decisión

- Implementación y pruebas de random forest

- Análisis de resultados

- Documentación y bitácora
