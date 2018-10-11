from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
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
        
        
        
        
    
        
        