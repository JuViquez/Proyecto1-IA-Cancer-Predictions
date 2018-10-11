import pandas as pd

def csv_to_dataset(file_path):
    return pd.read_csv(file_path)
    
def dataset_to_csv( file_path, dataset):
    dataset.to_csv(file_path, index = False)
     
def write_to_log( file_path, text):
    file = open(file_path, 'a')
    file.write(text)
    file.close()
        
    