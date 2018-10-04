import pytest

def test_l1_output():
    y_test = 2
    y_pred = 3
    assert Metrics.l1(y_test,y_pred) == 1
    
def test_l2_output():
    y_test = 2
    y_pred = 3
    assert Metrics.l2(y_test,y_pred) == (y_test - y_pred)**2
    
def test_l0_1_output_1():
    y_test = 2
    y_pred = 2
    assert Metrics.l0_1(y_test,y_pred) == 1

def test_l0_1_output_0():
    y_test = 3
    y_pred = 2
    assert Metrics.l0_1(y_test,y_pred) == 0
