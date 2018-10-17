import tensorflow as tf
import numpy as np
from source.models.Model import Model

class NeuralNetwork(Model):
    
    def __init__(self, num_layers, neurons_per_hidden_layer,
                 neurons_output_layer, activation_func = tf.nn.relu,
                 output_activation_func = tf.nn.softmax,
                 loss = 'binary_crossentropy', optimizer = 'adam', epochs = 5):
        
        super().__init__(size = num_layers)
        self.neurons_per_layer = neurons_per_hidden_layer
        self.activation_func = activation_func
        self.tf_model = tf.keras.models.Sequential()
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.neurons_output_layer = neurons_output_layer
        self.output_activation_func = output_activation_func

    def fit(self, X, Y):
        self.tf_model = tf.keras.models.Sequential()
        self.tf_model.add(tf.keras.layers.Flatten())
        for _ in range(self.size-1):
            self.tf_model.add(tf.keras.layers.Dense(self.neurons_per_layer, 
                                               activation = self.activation_func))
        self.tf_model.add(tf.keras.layers.Dense(self.neurons_output_layer, 
                                               activation = self.output_activation_func))
        self.tf_model.compile(optimizer=self.optimizer,
              loss = self.loss,
              metrics=['accuracy'])
        self.tf_model.fit(X, Y, epochs = self.epochs)
    
    def predict(self, X):
        return np.argmax(self.tf_model.predict([[X]]))  
        