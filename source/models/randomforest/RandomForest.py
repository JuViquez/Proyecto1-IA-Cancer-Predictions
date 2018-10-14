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

rf = RandomForest(2)
rf.fit(dataset,classification)
for i in rf.trees:
    i.root.print_tree(0)
    print("-----------------------------------------------")
print(rf.predict(['Yes', 'No', 'No', 'Yes', 'Full', '1', 'No', 'No', 'Thai', '60']))