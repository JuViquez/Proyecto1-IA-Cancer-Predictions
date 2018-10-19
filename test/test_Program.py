import pytest
from unittest.mock import Mock
from unittest.mock import patch
from source.datahandlers.FileManager import csv_to_dataset
from source.datahandlers.FileManager import dataset_to_csv

@pytest.fixture
def args(red_neuronal, arbol):
    args = { "prefijo": "test_dataset.csv",
            "indice_columna_y" : 0,
            "porcentaje_pruebas" : 0.25,
            "arbol" : arbol,
            "umbral_poda" : 0,
            "red_neuronal" : red_neuronal,
            "numero_capas" : 5,
            "unidades_por_capa" : 5,
            "funcion_activacion" : 'relu',
            "funcion_activacion_salida" : "softmax"
            }
    return args


