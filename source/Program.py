import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from source.datahandlers.DataManager import DataManager
from source.cross_validation.CrossValidationManager import CrossValidationManager
from source.datahandlers.FileManager import csv_to_dataset
from source.utilities.Constants import DATASETS_DIRECTORY
from source.models.neural_network.NeuralNetwork import NeuralNetwork
from source.utilities.Metrics import l0_1_loss

class Program:
    
    def __init__(self):
        self.data_manager = DataManager([])
        self.X = None
        self.y = None
        
    def main(self, args):
        dataset_path = DATASETS_DIRECTORY + args.prefijo
        dataset = csv_to_dataset(dataset_path)
        self.data_manager.dataset = dataset
        self.data_manager.shuffle_dataset()
        
        test_size = args.porcentaje_pruebas
        self.X, self.y = self.data_manager.preprocess_data(args.indice_columna_y)
        
        model_name = args.nombre_modelo
        
        if(args.arbol):
            prune_gain = args.umbral_poda    
        elif(args.red_neuronal):
            layers = args.numero_capas
            neurons_hidden_layer = args.unidades_por_capa
            activation_func = args.funcion_activacion
            output_activation_func = args.funcion_activacion_salida
    
    def create_random_forest(self, prune_gain):
        pass

    def create_neural_network(self, layers, neurons_hidden_layer,
                              activation_func, output_activation_func,
                              test_size):
        
        label_encoder = self.data_manager.create_encoder(self.y)
        self.y = label_encoder.transform(self.y)
        X_train, y_train, X_test, y_test = self.data_manager.split_train_test(self.X, self.y, test_size)
        activation_func = self.choose_activation_function(activation_func)
        output_activation_func = self.choose_activation_function(output_activation_func)
        
        neurons_output_layer = len(label_encoder.classes_)
        neural_network = NeuralNetwork(layers, neurons_hidden_layer,
                                       neurons_output_layer, activation_func,
                                       output_activation_func)
        
        cvm = CrossValidationManager(neural_network, X_train, y_train, l0_1_loss)
        err_t, _ = cvm.cross_validation()
        err_v = cvm.error_rate(X_test, y_test)
        
        predictions = predictions_list(cvm.learner, self.X)
        predictions = label_encoder.inverse_transform(predictions)
        self.data_manager.add_column(predictions, column_name = 'predictions' )
            
    def choose_activation_function(self,activation_func):
        if activation_func == 'relu':
            return tf.nn.relu
        if activation_func == 'softmax':
            return tf.nn.softmax
        if activation_func == 'softplus':
            return tf.nn.softplus
        if activation_func == 'sigmoid':
            return tf.nn.sigmoid
        raise ValueError(activation_function + 'is not an activation function or is not implemented yet')
    
    def predictions_list(learner, X):
        predictions = []
        for x in X:
            prediction = learner.predict(x)
            predictions.append(prediction)
        return predictions