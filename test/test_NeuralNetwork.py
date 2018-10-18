import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock
from unittest.mock import patch
from source.models.neural_network.NeuralNetwork import NeuralNetwork

@pytest.fixture
def neural_network():
    return NeuralNetwork(5, 5, 3,tf.nn.relu, tf.nn.softmax, 'binary_crossentropy', 'sgd', 2)

@patch('tensorflow.keras.models.Sequential', return_value = Mock())
@patch('tensorflow.keras.layers.Dense', return_value = None)
@patch('tensorflow.keras.layers.Flatten', return_value = None)
def test_fit_tf_parameters(mock_flatten, mock_dense, mock_sequential):
    nw = neural_network()
    x = np.array([
            [1,2], 
            [3,4]
        ])
    y = np.array([5,6])
    
    nw.fit(x,y)    
    nw.tf_model.add.assert_called_with(mock_flatten())
    nw.tf_model.add.assert_called_with(mock_dense(5, tf.nn.relu))
    nw.tf_model.add.assert_called_with(mock_dense(2, tf.nn.softmax))
    nw.tf_model.compile.assert_called_with(optimizer = 'sgd', 
                                          loss='binary_crossentropy', 
                                          metrics = ['accuracy'])
    nw.tf_model.fit.assert_called_once_with(x,y,epochs = 2)

def test_predict_tf_parameters():
    nw = neural_network()
    nw.tf_model = Mock()
    nw.predict([1])
    nw.tf_model.predict.assert_called_with([[1]])