import math
from DecisionTreeNode import DecisionTreeNode
from Leaf import Leaf


class Node(DecisionTreeNode):
    def __init__(self,
                subdataset,
                classification):
        super().__init__(None,0)
        self.subdataset = subdataset
        self.classification = classification
        self.branch = []
        self.gain = 0
        
    def count_rows(self,array_type,column):
        counter = {}
        if isinstance(array_type[0], list):
            for row in array_type:
                tag = row[column]
                if tag not in counter:
                    counter[tag] = 0
                counter[tag] += 1
        else:
            for element in array_type:
                if element not in counter:
                    counter[element] = 0
                counter[element] += 1
        return counter

    def entropy(self,probability):
        entropy = 0
        if(probability != 0 and probability != 1):
            entropy = -1 * (probability*math.log2(probability) + (1 - probability)*math.log2(1 - probability))
        return entropy

    def remainder(self,column):
        probability = 0
        remainder_return = 0
        for key,value in self.count_rows(self.subdataset,column).items():
            probability = value / len(self.classification)
            remainder_return =+ probability*self.entropy(probability)
        return remainder_return

    def best_gain(self):
        num = next(iter(self.count_rows(self.classification,0).values())) / len(self.classification)
        dataset_entropy = self.entropy(num)
        for i in range(len(self.subdataset[0])):
            column_gain = dataset_entropy - self.remainder(i)
            if column_gain > self.gain:
                self.gain = column_gain
                self.column = i

    def split_dataset(self,value):
        array_return = []
        classification_return = []
        subdataset_copy = self.subdataset.copy()
        for i in range(len(subdataset_copy)):
            if subdataset_copy[i][self.column] == value:
                del subdataset_copy[i][self.column]
                array_return.append(subdataset_copy[i])
                classification_return.append(self.classification[i])
        return array_return,classification_return

    def plurality(self,prediction):
        default = {'Default':0}
        for tag in prediction:
            if (prediction[tag]>=default[list(default)[0]]):
                default = {tag : prediction[tag]}
        return default

    def tree_learning(self,num):
        if(len(self.subdataset) == 0):
            return 1
        classification_values = self.count_rows(self.classification,0)
        difnum =+ num + 1
        if (len(classification_values) < 2 or len(self.subdataset[0]) == 0):
            return classification_values
        
        self.best_gain()

        if(self.gain == 0):
            print("Ganancia cero")
            return 4

        column_values = self.count_rows(self.subdataset,self.column)
        for key in column_values:
            dat,clas = self.split_dataset(key)
            tree_node = Node(dat.copy(),clas.copy())
            tree_node.question = key
            self.branch.append(tree_node)
            del tree_node #probar
        for i in range(len(self.branch)):
            response = self.branch[i].tree_learning(difnum)
            if isinstance(response,dict):
                self.branch[i] = Leaf(self.plurality(response),self.branch[i].question,self.column)
            elif response == 1:
                dictionary = self.count_rows(self.classification,0)
                self.branch[i] = Leaf(self.plurality(dictionary),self.question,self.column)
        return 0

    def print_tree(self,num):
        num += 1
        for node in self.branch:
            print('  '*num + node.question+"  column: "+str(self.column))
            node.print_tree(num)

    def predict(self,row):
        for node in self.branch:
            if node.question == row[self.column]:
                del row[self.column]
                print(row)
                return node.predict(row)
        
               
columnas = ['Tipo', 'Curso', 'Interes', 'Tiempo Disponible', 'Conocimiento','Clasificador']

#classification = ['Si','Si','No','Si','Si','No','No','Si','No','Si','Si','No','No','Si','No','Si','Si','Si','Si','Si','No']
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
dataset2 = [
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


nodo = Node(dataset[:],classification)
nodo.tree_learning(0)
nodo.print_tree(0)
print(dataset2)
for i in range(len(dataset2)):
    print(dataset2[i])
    print(nodo.predict(dataset2[i]))
    print(classification[i])
    print("--------------------------------------------------")
