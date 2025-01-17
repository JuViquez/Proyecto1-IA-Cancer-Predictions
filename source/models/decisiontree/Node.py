import math
from random import randint
import numpy as np
from source.models.decisiontree.DecisionTreeNode import DecisionTreeNode
from source.models.decisiontree.Leaf import Leaf


class Node(DecisionTreeNode):
    def __init__(self):
        super().__init__(None, 0)
        self.branch = []
        self.gain = 0

    def count_rows(self, array_type, column):
        counter = {}
        if isinstance(array_type[0], str):
            for element in array_type:
                if element not in counter:
                    counter[element] = 0
                counter[element] += 1
        else:
            for row in array_type:
                tag = row[column]
                if tag not in counter:
                    counter[tag] = 0
                counter[tag] += 1

        return counter

    def count_probability(self, subdataset, classification, column, key):
        value = classification[0]
        return_probability = 0
        for i in range(len(classification)):
            if(subdataset[i][column] == key and classification[i] == value):
                return_probability += 1
        return return_probability

    def entropy(self, probability):
        entropy = 0
        if(probability != 0 and probability != 1):
            entropy = -1 * (probability * math.log2(probability) +
                            (1 - probability) * math.log2(1 - probability))
        return entropy

    def remainder(self, subdataset, classification, column):
        probability = 0
        remainder_return = 0
        for key, value in self.count_rows(subdataset, column).items():
            probability = value / len(classification)
            key_probability = self.count_probability(
                subdataset, classification, column, key) / value
            remainder_return += probability * self.entropy(key_probability)
        return remainder_return

    def split_features(self, size):
        features = []
        feature_size = int(math.sqrt(size))
        while feature_size > 0:
            random_feature = randint(0, size - 1)
            if random_feature not in features:
                features.append(random_feature)
                feature_size = feature_size - 1
        return features

    def best_gain(self, subdataset, classification):
        num = next(
            iter(
                self.count_rows(
                    classification,
                    0).values())) / len(classification)
        dataset_entropy = self.entropy(num)
        features = self.split_features(len(subdataset[0]))
        for i in features:
            column_gain = dataset_entropy - \
                self.remainder(subdataset, classification, i)
            if column_gain > self.gain:
                self.gain = column_gain
                self.column = i

    def plurality(self, prediction):
        if not prediction:
            return {}
        default = {'Default': 0}
        for tag in prediction:
            if (prediction[tag] >= default[list(default)[0]]):
                default = {tag: prediction[tag]}
        return default
