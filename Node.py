import math
from Leaf import Leaf

class Node():
    def __init__(self,
                subdataset,
                classification):
        self.subdataset = subdataset
        self.classification = classification
        self.branch = []
        self.column = 0
        self.gain = 0
        self.question = None
        
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
        entropy = probability*math.log2(probability) + (1 - probability)*math.log2(1 - probability)
        return entropy * -1

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
        classification_values = self.count_rows(self.classification,0)
        difnum =+ num + 1
        if (len(classification_values) < 2):
            return classification_values
        if(len(self.subdataset) == 0):
            return 1
        if(len(self.subdataset[0]) == 0):
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
                self.branch[i] = Leaf(self.plurality(response),self.branch[i].question)
        return 0
        
               
columnas = ['Tipo', 'Curso', 'Interes', 'Tiempo Disponible', 'Conocimiento','Clasificador']

classification = ['Si','Si','No','Si','Si','No','No','Si','No','Si','Si','No','No','Si','No','Si','Si','Si','Si','Si','No']

dataset = [
        ["Proyecto corto", 'IA', 'C', 'Alto', 'M'],
        ["Examen", 'IA', 'L', 'Alto', 'B'],
        ["Tarea", 'Seminario','S', 'Bajo', 'A'],
        ["Proyecto", 'AP', 'S', 'Medio', 'A'],
        ["Tarea", 'Seminario', 'L', 'Alto', 'M'],
        ["Proyecto corto", 'AP', 'C', 'Medio', 'B'],
        ["Examen", 'Seminario', 'S', 'Medio', 'A'],
        ["Proyecto", 'AP', 'S', 'Alto', 'B'],
        ["Proyecto", 'Redes', 'C', 'Medio', 'M'],
        ["Examen", 'Seminario', 'S', 'Bajo', 'M'],
        ["Proyecto corto", 'IA', 'S', 'Alto', 'B'],
        ["Examen", 'Seminario', 'L', 'Medio', 'A'],
        ["Tarea", 'AP', 'C', 'Bajo', 'B'],
        ["Proyecto", 'IA', 'C', 'Medio', 'M'],
        ["Tarea", 'Redes', 'L', 'Medio', 'M'],
        ["Examen", 'Redes', 'L', 'Bajo', 'A'],
        ["Proyecto", 'Seminario', 'L', 'Bajo', 'A'],
        ["Proyecto", 'IA', 'S', 'Medio', 'M'],
        ["Tarea", 'AP', 'C', 'Medio', 'B'],
        ["Tarea", 'IA', 'C', 'Medio', 'B'],
        ["Examen", 'Seminario', 'S', 'Medio', 'B'],
]
nodo = Node(dataset,classification)
data1 = [1,2,3]
data2 = data1.copy()
data2.pop(1)
print(data1)
nodo.tree_learning(0)
