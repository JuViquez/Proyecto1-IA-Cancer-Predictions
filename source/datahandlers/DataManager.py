import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from source.exceptions.EmptyDataset import EmptyDataset

class DataManager:
    
    def __init__(self, dataset):
        self.dataset = pd.DataFrame(dataset)
    
    def split_X_y(self, y_index):
        values = self.dataset.iloc[ :, : ].values
        y = values[: , y_index]
        X = np.delete(values, y_index, axis = 1)
        return X, y
    
    def add_column(self, column, column_name):
        column = pd.DataFrame({column_name : column})
        self.dataset = pd.concat([self.dataset,column], axis = 1)
        
    def preprocess_data(self, y_index):
        if self.dataset.empty:
            raise EmptyDataset("Cannot preprocess an empty dataset")
        X, y = self.split_X_y(y_index)
        sc = StandardScaler()
        X = sc.fit_transform(X)
        return X, y
    
    def shuffle_dataset(self):
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
    
    def create_encoder(self, labels):
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder

    def split_train_test(self, X, y, test_size = 0.25):
        if(test_size < 0 or test_size > 1):
            raise ValueError("test_size must be a number between 0 and 1")
        size = round(len(X) * test_size)
        X_test = X[0:size]
        y_test = X[0:size]
        del_range = list(range(size))
        X_train = np.delete(X, del_range, axis = 0)
        y_train = np.delete(y, del_range, axis = 0)
        return X_train, y_train, X_test, y_test
        
        
        
        
        
    
        
        