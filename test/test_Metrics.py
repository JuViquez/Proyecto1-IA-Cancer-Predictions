import pytest
from source.utilities.Metrics import *

def test_l1_output():
    y_test = 2
    y_pred = 3
    assert l1_loss(y_test,y_pred) == 1
    
def test_l2_output():
    y_test = 2
    y_pred = 3
    assert l2_loss(y_test,y_pred) == (y_test - y_pred)**2
    
def test_l0_1_output_1():
    y_test = 2
    y_pred = 2
    assert l0_1_loss(y_test,y_pred) == 0

def test_l0_1_output_0():
    y_test = 3
    y_pred = 2
    assert l0_1_loss(y_test,y_pred) == 1
