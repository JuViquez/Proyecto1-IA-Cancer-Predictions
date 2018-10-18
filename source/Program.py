import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from source.datahandlers.DataManager import DataManager
from source.cross_validation.CrossValidationManager import CrossValidationManager
from source.datahandlers.FileManager import csv_to_dataset
from source.datahandlers.FileManager import dataset_to_csv
from source.utilities.Constants import DATASETS_DIRECTORY
from source.models.neural_network.NeuralNetwork import NeuralNetwork
from source.utilities.Metrics import l0_1_loss

class Program:
    
    def __init__(self):
        self.data_manager = DataManager([])
        self.X = None
        self.y = None
        
    def main(self, args):
        dataset_path = DATASETS_DIRECTORY + '/' + args.prefijo
        print('opening ' + dataset_path)
        dataset = csv_to_dataset(dataset_path)
        self.data_manager.dataset = dataset
        self.data_manager.shuffle_dataset()
        
        test_size = args.porcentaje_pruebas
        self.X, self.y = self.data_manager.preprocess_data(args.indice_columna_y)
        
        err_t = 0
        err_v = 0
        
        if(args.arbol):
            prune_gain = args.umbral_poda
            
        elif args.red_neuronal: #neural network
            layers = args.numero_capas
            neurons_hidden_layer = args.unidades_por_capa
            activation_func = args.funcion_activacion
            output_activation_func = args.funcion_activacion_salida
            err_t, err_v = self.process_neural_network(layers, neurons_hidden_layer,
                                                 activation_func, output_activation_func,
                                                 test_size)
            prediction_path = DATASETS_DIRECTORY + '/neural_network_predictions_' + args.prefijo
            dataset_to_csv(prediction_path, self.data_manager.dataset)
           
        print('Error de entrenamiento: ' + str(err_t) + 
              '\n' + 'Error de pruebas: ' + str(err_v) + '\n')
        
    
    def create_random_forest(self, prune_gain):
        pass

    def process_neural_network(self, layers, neurons_hidden_layer,
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
        
        predictions = self.predictions_list(cvm.learner, self.X)
        predictions = label_encoder.inverse_transform(predictions)
        self.data_manager.add_column(predictions, column_name = 'predictions' )
        
        return err_t, err_v
            
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
    
    def predictions_list(self,learner, X):
        predictions = []
        for x in X:
            prediction = learner.predict(x)
            predictions.append(prediction)
        return predictions