import pytest
from source.models.decisiontree.Leaf import Leaf
from source.models.decisiontree.Node import Node
from source.models.decisiontree.DecisionTree import DecisionTree

ex_dt = DecisionTree()
classification = ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','No','No','Yes']
dataset = [
    ['Yes', 'No', 'No', 'Yes', 'Some', '3', 'No', 'Yes', 'French', '10'],
    ['Yes', 'No', 'No', 'Yes', 'Full', '1', 'No', 'No', 'Thai', '60'],
    ['No', 'Yes', 'No', 'No', 'Some', '1', 'No', 'No', 'Burger', '10'],
    ['Yes', 'No', 'Yes', 'Yes', 'Full', '1', 'Yes', 'No', 'Thai', '30'],
    ['Yes', 'No', 'Yes', 'No', 'Full', '3', 'No', 'Yes', 'French', '80'],
    ['No', 'Yes', 'No', 'Yes', 'Some', '2', 'Yes', 'Yes', 'Italian', '10'],
    ['No', 'Yes', 'No', 'No', 'None', '1', 'Yes', 'No', 'Burger', '10'],
    ['No', 'No', 'No', 'Yes', 'Some', '2', 'Yes', 'Yes', 'Thai', '10'],
    ['No', 'Yes', 'Yes', 'No', 'Full', '1', 'Yes', 'No', 'Burger', '80'],
    ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '3', 'No', 'Yes', 'Italian', '30'],
    ['No', 'No', 'No', 'No', 'None', '1', 'No', 'No', 'Thai', '10'],
    ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '1', 'No', 'No', 'Burger', '60']
]

def test_fit():
    ex_dt.fit(dataset,classification)
    assert round(ex_dt.root.gain,2) == 0.54

def test_split_dataset_empty():
    dataset, classification = ex_dt.split_dataset([],[],'0',0)
    assert not dataset and not classification

def test_split_dataset():
    dataset = [
        ['T','0'],['F','0'],['T','0'],
        ['F','0'],['T','0'],['F','0'],
        ['T','1'],['F','1'],['T','1'],
        ['F','1'],['T','1'],['F','1'],
    ]
    classification = [
        'Yes','Yes','Yes',
        'Yes','Yes','Yes',
        'No','No','No',
        'No','No','No'
    ]
    result_dataset =[
        ['T'],['F'],['T'],
        ['F'],['T'],['F']
    ]
    result_class = [
        'Yes','Yes','Yes',
        'Yes','Yes','Yes'
    ]
    dataset, classification = ex_dt.split_dataset(dataset,classification,'0',1)
    assert dataset == result_dataset and classification == result_class

def test_prune():
    ex_dt.fit(dataset,classification)
    ex_dt.prune(0.2)
    assert isinstance(ex_dt.root.branch[1],Leaf) 

def test_prune_1():
    ex_dt.fit(dataset,classification)
    ex_dt.prune(1)
    assert isinstance(ex_dt.root.branch[1],Leaf) 

def test_prune_0():
    ex_dt.fit(dataset,classification)
    ex_dt.prune(0.0)
    assert isinstance(ex_dt.root.branch[1],Node)

def test_predict_No():
    ex_dt.fit(dataset,classification)
    X = ['Yes', 'No', 'No', 'Yes', 'Full', '1', 'No', 'No', 'Thai', '60']
    assert ex_dt.predict(X) == 'No'

def test_predict_Yes():
    ex_dt.fit(dataset,classification)
    X = ['No', 'Yes', 'No', 'No', 'Some', '1', 'No', 'No', 'Burger', '10']
    assert ex_dt.predict(X) == 'Yes'