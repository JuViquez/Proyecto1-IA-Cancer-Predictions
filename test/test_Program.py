import pytest
import tensorflow as tf
import numpy as np
from argparse import Namespace
from unittest.mock import Mock
from source.Program import Program


@pytest.fixture
def args_fixture(red_neuronal, arbol):
    args = Namespace(prefijo="test_dataset.csv",
                     indice_columna_y=0,
                     porcentaje_pruebas=0.25,
                     arbol=arbol,
                     umbral_poda=0,
                     red_neuronal=red_neuronal,
                     numero_capas=5,
                     unidades_por_capa=5,
                     funcion_activacion='relu',
                     funcion_activacion_salida="softmax",
                     iteraciones_optimizador=10
                     )
    return args


def test_main_neural_network():
    args = args_fixture(True, False)
    p = Program()
    p.data_manager = Mock()
    p.data_manager.preprocess_data = Mock(return_value=[[2], [3]])
    p.process_neural_network = Mock(return_value=[2, 3])

    p.main(args)
    p.data_manager.shuffle_dataset.assert_called_with()
    p.data_manager.preprocess_data.assert_called_with(args.indice_columna_y)

    p.process_neural_network.assert_called_with(
        args.numero_capas,
        args.unidades_por_capa,
        args.funcion_activacion,
        args.funcion_activacion_salida,
        args.porcentaje_pruebas,
        args.iteraciones_optimizador)


def test_main_random_forest():
    args = args_fixture(False, True)
    p = Program()
    p.data_manager = Mock()
    p.data_manager.preprocess_data = Mock(return_value=[[2], [3]])
    p.create_random_forest = Mock(return_value=[2, 3])

    p.main(args)
    p.data_manager.shuffle_dataset.assert_called_with()
    p.data_manager.preprocess_data.assert_called_with(args.indice_columna_y)

    p.create_random_forest.assert_called_with(
        args.umbral_poda, args.porcentaje_pruebas)


def test_main_value_error():
    p = Program()
    args = args_fixture(True, True)
    with pytest.raises(ValueError):
        p.main(args)
    args = args_fixture(False, False)
    with pytest.raises(ValueError):
        p.main(args)


def test_choose_activation_function_output():
    p = Program()
    assert tf.nn.relu == p.tf_activation_function("relu")
    assert tf.nn.softmax == p.tf_activation_function("softmax")
    assert tf.nn.softplus == p.tf_activation_function("softplus")
    assert tf.nn.sigmoid == p.tf_activation_function("sigmoid")


def test_choose_activation_function_value_error():
    p = Program()
    with pytest.raises(ValueError):
        p.tf_activation_function('elu')


def test_prediction_list():
    p = Program()
    learner = Mock()
    learner.predict.return_value = 1
    X = [2, 3]
    result = p.predictions_list(learner, X)
    expected_result = [1, 1]
    assert np.array_equal(result, expected_result)
