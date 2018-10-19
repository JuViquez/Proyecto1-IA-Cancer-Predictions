import pytest
from source.models.decisiontree.Node import Node

example_node = Node()

def test_count_rows_2d_array():
    dataset = [
        ['1'], ['0'], 
        ['1'], ['1'],
        ['0'], ['1'], 
        ['0'], ['1']
    ]
    result = {}
    result['0'] = 3
    result['1'] = 5
    assert example_node.count_rows(dataset,0) == result

def test_count_rows_list():
    dataset = [
        'Yes','No','No',
        'No','Yes','No',
        'Yes','No','Yes'
    ]
    result = {}
    result['Yes'] = 4
    result['No'] = 5
    assert example_node.count_rows(dataset,0) == result

def test_count_probability():
    dataset = [
        ['1'], ['0'], 
        ['1'], ['1'],
        ['0'], ['1'], 
        ['0'], ['1']
    ]
    classification = [
        'Yes','Yes','No','Yes',
        'Yes','No','No','No'
    ]
    assert example_node.count_probability(dataset,classification,0,'1') == 2

def test_entropy_50():
    assert example_node.entropy(0.5) == 1

def test_entropy_100():
    assert example_node.entropy(1) == 0

def test_remainder():
    dataset = [
        ['1'],['1'],['1'],
        ['0'],['0'],['0'],
        ['1'],['1'],['0'],
        ['1'],['0'],['0'],
    ]
    classification = [
        'Yes','No','No',
        'No','Yes','No',
        'Yes','No','Yes',
        'Yes','No','Yes'
    ]
    assert example_node.remainder(dataset,classification,0) == 1

def test_best_gain_value():
    example_node.gain = 0
    example_node.column = 0
    dataset = [
        ['0'],['0'],['0'],
        ['0'],['0'],['0'],
        ['1'],['1'],['1'],
        ['1'],['1'],['1'],
    ]
    classification = [
        'Yes','Yes','Yes',
        'Yes','Yes','Yes',
        'No','No','No',
        'No','No','No'
    ]
    example_node.best_gain(dataset,classification)
    assert round(example_node.gain,2) == 1

def test_best_gain_column():
    example_node.gain = 0
    example_node.column = 0
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
    example_node.best_gain(dataset,classification)
    assert example_node.column == 1

def test_plurality_empty():
    assert not example_node.plurality({})

def test_plurality():
    prediction = {"Yes":5,"No":7}
    assert example_node.plurality(prediction) == {"No":7}