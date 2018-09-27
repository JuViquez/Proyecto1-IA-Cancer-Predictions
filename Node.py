import math

class Node():
    def __init__(self,
                subdataset,
                classification):
        self.subdataset = subdataset
        self.classification = classification
        self.branch = []
        self.column = 0
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
        for i in range(len(self.subdataset)):
            if self.subdataset[i][self.column] == value:
                self.subdataset[i].pop(self.column)
                array_return.append(self.subdataset[i])
                classification_return.append(self.classification[i])
        return array_return,classification_return

    def tree_learning(self):
        classification_values = self.count_rows(self.classification,0)
        if (len(classification_values) < 2):
            print(classification_values)
            return 0
        if(len(self.subdataset) == 0):
            #return plurality of father
            print("Pluralidad pdre")
            return 0
        if(len(self.subdataset[0]) == 0):
            print("pluralidad")
            print (classification_values)
            return 0

        self.best_gain()
        if(self.gain == 0):
            print("Ganancia cero")
            return 0

        column_values = self.count_rows(self.subdataset,self.column)
        for key in column_values:
            dat,clas = self.split_dataset(key)
            tree_node = Node(dat,clas)
            print("----------------")
            tree_node.tree_learning()
            self.branch.append(tree_node)
        
               


classification = ['Si','Si','No','Si','Si','No','No','Si','No','Si','Si','No','No','Si','No','Si','Si','Si','Si','Si','No']

dataset = [
        ["Proyecto corto", 'IA', 'Medio', 'Alto', 'Medio','Si'],
        ["Examen", 'IA', 'Alto', 'Alto', 'Bajo','Si'],
        ["Tarea", 'Seminario','Medio', 'Bajo', 'Alto','No'],
        ["Proyecto", 'AP', 'Bajo', 'Medio', 'Alto','Si'],
        ["Tarea", 'Seminario', 'Alto', 'Alto', 'Medio','Si'],
        ["Proyecto corto", 'AP', 'Medio', 'Medio', 'Bajo','No'],
        ["Examen", 'Seminario', 'Bajo', 'Medio', 'Alto', 'No'],
        ["Proyecto", 'AP', 'Bajo', 'Alto', 'Bajo','Si'],
        ["Proyecto", 'Redes', 'Medio', 'Medio', 'Medio', 'No'],
        ["Examen", 'Seminario', 'Bajo', 'Bajo', 'Medio', 'Si'],
        ["Proyecto corto", 'IA', 'Bajo', 'Alto', 'Bajo', 'Si'],
        ["Examen", 'Seminario', 'Alto', 'Medio', 'Alto', 'No'],
        ["Tarea", 'AP', 'Medio', 'Bajo', 'Bajo','No'],
        ["Proyecto", 'IA', 'Medio', 'Medio', 'Medio', 'Si'],
        ["Tarea", 'Redes', 'Alto', 'Medio', 'Medio','No'],
        ["Examen", 'Redes', 'Alto', 'Bajo', 'Alto', 'Si'],
        ["Proyecto", 'Seminario', 'Alto', 'Bajo', 'Alto', 'Si'],
        ["Proyecto", 'IA', 'Bajo', 'Medio', 'Medio','Si'],
        ["Tarea", 'AP', 'Medio', 'Medio', 'Bajo','Si'],
        ["Tarea", 'IA', 'Medio', 'Medio', 'Bajo','Si'],
        ["Examen", 'Seminario', 'Bajo', 'Medio', 'Bajo','No'],
]
nodo = Node(dataset,classification)

nodo.tree_learning()
