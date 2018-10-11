import pytest
from source.decisiontree.Node import Node

def test_count_rows_2d_array():
    example_dataset = [
        ['1'], ['0'], 
        ['1'], ['1'],
        ['0'], ['1'], 
        ['0'], ['1']
    ]
    example_node = Node()
    result = {}
    result['0'] = 3
    result['1'] = 5
    assert example_node.count_rows(example_dataset,0) == result

def test_count_rows_list():
    example_dataset = [
        'Yes','No','No',
        'No','Yes','No',
        'Yes','No','Yes'
    ]
    example_node = Node()
    result = {}
    result['Yes'] = 4
    result['No'] = 5
    assert example_node.count_rows(example_dataset,0) == result

def test_entropy_50():
    example_node = Node()
    assert example_node.entropy(0.5) == 1

def test_entropy_100():
    example_node = Node()
    assert example_node.entropy(1) == 0

def test_remainder():
    example_dataset = [
        ['1'],['1'],['1'],
        ['0'],['0'],['0'],
        ['1'],['1'],['0'],
        ['1'],['0'],['0'],
    ]
    example_classification = [
        'Yes','No','No',
        'No','Yes','No',
        'Yes','No','Yes',
        'Yes','No','Yes'
    ]
    example_node = Node()
    assert example_node.remainder(example_dataset,example_classification,0) == 1
