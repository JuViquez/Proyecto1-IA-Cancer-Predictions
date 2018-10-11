import math
import copy
from DecisionTreeNode import DecisionTreeNode
from Leaf import Leaf


class Node(DecisionTreeNode):
    def __init__(self):
        super().__init__(None,0)
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

    def remainder(self,subdataset,classification,column):
        probability = 0
        remainder_return = 0
        save_column = self.column
        for key,value in self.count_rows(subdataset,column).items():
            probability = value / len(classification)
            self.column = column
            value_dataset, value_classification = self.split_dataset(subdataset,classification,key)
            sub_rows = self.count_rows(value_classification,0)
            sub_probability = next(iter(sub_rows.values())) / value
            remainder_return += probability*self.entropy(sub_probability)            
        self.column = save_column
        return remainder_return

    def best_gain(self,subdataset,classification):
        num = next(iter(self.count_rows(classification,0).values())) / len(classification)
        dataset_entropy = self.entropy(num)
        for i in range(len(subdataset[0])):
            column_gain = dataset_entropy - self.remainder(subdataset,classification,i)
            if column_gain > self.gain:
                self.gain = column_gain
                self.column = i

    def split_dataset(self,subdataset,classification,value):
        array_return = []
        classification_return = []
        subdataset_copy = copy.deepcopy(subdataset)
        for i in range(len(subdataset_copy)):
            if subdataset_copy[i][self.column] == value:
                del subdataset_copy[i][self.column]
                array_return.append(subdataset_copy[i])
                classification_return.append(classification[i])
        return array_return,classification_return

    def plurality(self,prediction):
        default = {'Default':0}
        for tag in prediction:
            if (prediction[tag]>=default[list(default)[0]]):
                default = {tag : prediction[tag]}
        return default

    def tree_learning(self,subdataset,classification,num):
        
        if(len(subdataset) == 0):
            return 1

        classification_values = self.count_rows(classification,0)
        difnum =+ num + 1
        if (len(classification_values) < 2 or len(subdataset[0]) == 0):
            return classification_values
        
        self.best_gain(subdataset,classification)
        
        if(self.gain == 0):
            print("Ganancia cero - error")
            return 1

        column_values = self.count_rows(subdataset,self.column)
        for key in column_values:
            dat,clas = self.split_dataset(subdataset,classification,key)
            tree_node = Node()
            tree_node.question = key
            response = tree_node.tree_learning(dat,clas,difnum)
            if isinstance(response,dict):
                leaf_node =  Leaf(self.plurality(response),tree_node.question,self.column)
                self.branch.append(leaf_node)
            elif response == 1:
                dictionary = self.count_rows(classification,0)
                leaf_node = Leaf(self.plurality(dictionary),self.question,self.column)
                self.branch.append(leaf_node)
            else: 
                self.branch.append(tree_node)
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

nodo = Node()
nodo.tree_learning(dataset,classification,0)
nodo.print_tree(0)
print(dataset)

#print(dataset2)
for i in range(len(dataset)):
    print(dataset[i])
    print(nodo.predict(dataset[i]))
    print(classification[i])
    print("--------------------------------------------------")
