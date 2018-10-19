from random import randint
import copy
from source.models.Model import Model
from source.models.decisiontree.DecisionTree import DecisionTree

class RandomForest(Model):
    def __init__(self,
                size):
        super().__init__(size)
        self.trees = []
    
    def fit(self,X,Y):
        dataset_size = round(len(X)/self.size)
        for i in range(self.size):
            subdataset, classification = self.split_dataset(X,Y,dataset_size)
            tree = DecisionTree()
            self.trees.append(tree)
            tree.fit(subdataset,classification)
    
    def predict(self,X):
        results = {}
        for node in self.trees:
            tag = node.predict(X)
            if tag not in results:
                results[tag] = 0
            results[tag] += 1
        print(results)
        return max(results, key = results.get)

    def split_dataset(self,X,Y,size):
        dataset = []
        classification = []
        for i in range(size):
            random_number = randint(0, len(X)-1)
            dataset.append(copy.deepcopy(X[random_number]))
            classification.append(copy.deepcopy(Y[random_number]))
        return dataset, classification