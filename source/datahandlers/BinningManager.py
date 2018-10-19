import numpy as np
from numbers import Number
from source.utilities.Constants import NUMERIC_RANGE as NR

class BinningManager():
    def __init__(self):
        self.slices = []

    def convert_data(self, X, column):
        for row in X:
            value = row[column]
            if value <= self.slices[-1][0]:
                row[column] = str(self.slices[-1][0])+"-"+str(self.slices[-1][1])
            elif value >= self.slices[-1][-1]:
                row[column] = str(self.slices[-1][-2])+"-"+str(self.slices[-1][-1])
            else:
                for i in range(len(self.slices[-1])-1):
                    if value >= self.slices[-1][i] and value < self.slices[-1][i+1]:
                        row[column] = str(self.slices[-1][i])+" - "+str(self.slices[-1][i+1])
                        continue

    def binning_data(self, X):
        for column in range(len(X[0])):
            if isinstance(X[0][column],Number):
                value_list = [i[column] for i in X]
                max_value = max(value_list)
                min_value = min(value_list)
                chunk = np.linspace(min_value, max_value, NR+1, True).astype(np.float)
                chunk = [round(x,2) for x in chunk]
                self.slices.append(chunk)
                self.convert_data(X,column)
"""
bm = BinningManager()
bm.binning_data([[0.58,2],[0.15,7],[0.25,8],[0.47,13],[0.95,24],[0.69,18],[0.29,3],[0.45,6],[0.78,9]])
"""