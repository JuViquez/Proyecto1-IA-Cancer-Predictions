from source.datahandlers.DataManager import DataManager
from source.datahandlers.FileManager import csv_to_dataset
from source.utilities.Constants import DATASETS_DIRECTORY

class Program:
    
    def __init__():
        self.data_manager = DataManager([])
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def main(self, args):
        dataset_path = DATASETS_DIRECTORY + args.prefijo
        dataset = csv_to_dataset(dataset_path)
        self.data_manager.dataset = dataset
        self.data_manager.shuffle_dataset()
        
        test_size = args.porcentaje_pruebas
        X, y = self.data_manager.preprocess_data(args.indice_columna_y)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_manager.split_train_test(X, y, test_size)
        
        model_name = args.nombre_modelo
        if(args.arbol):
            prune_gain = args.umbral_poda
        elif(args.red_neuronal):
            layers = args.numero_capas
            neurons_hidden_layer = args.unidades_por_capa
            activation_func = args.funcion_activacion
    
    def create_random_forest():
        pass

    def create_neural_network()