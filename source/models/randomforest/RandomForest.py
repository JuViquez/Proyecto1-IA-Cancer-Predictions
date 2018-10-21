from random import randint
import copy
from source.models.Model import Model
from source.models.decisiontree.DecisionTree import DecisionTree


class RandomForest(Model):
    def __init__(self,
                 size):
        super().__init__(size)
        self.trees = []

    def fit(self, X, Y):
        X = X.tolist()
        Y = Y.tolist()
        self.output = ""
        self.trees = []
        if self.size > 0:
            #dataset_size = round(len(X)/self.size)
            dataset_size = len(X)
            for i in range(self.size):
                subdataset, classification = self.split_dataset(
                    X, Y, dataset_size)
                tree = DecisionTree()
                self.trees.append(tree)
                self.output += "Tree: " + str(i) + " \n"
                tree.fit(subdataset, classification)
                tree.print_tree(tree.root, 0)
                self.output += tree.output + "\n \n"

    def predict(self, X):
        results = {}
        if not isinstance(X, list):
            X = X.tolist()
        for node in self.trees:
            tag = node.predict(X)
            if tag not in results:
                results[tag] = 0
            results[tag] += 1
        maximo = max(results, key=results.get)
        if maximo is None:
            del results[None]
            if not results:
                maximo = "?"
            else:
                maximo = max(results, key=results.get)
        return maximo

    def split_dataset(self, X, Y, size):
        dataset = []
        classification = []
        for i in range(size):
            random_number = randint(0, len(X) - 1)
            dataset.append(copy.deepcopy(X[random_number]))
            classification.append(copy.deepcopy(Y[random_number]))
        return dataset, classification
