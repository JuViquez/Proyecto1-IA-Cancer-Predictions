import copy
from source.decisiontree.Node import Node
from source.decisiontree.Leaf import Leaf

class DecisionTree():
    def __init__(self):
        self.root = None
        self.pruning_ratio = 0
    
    def fit(self,X,Y):
        self.root = Node()
        self.tree_learning(self.root,X,Y)

    def tree_learning(self,node,subdataset,classification):
        if(len(subdataset) == 0):
            return 1
        classification_values = node.count_rows(classification,0)
        if (len(classification_values) < 2 or len(subdataset[0]) == 0):
            return classification_values
        node.best_gain(subdataset,classification)
        if(node.gain == 0):
            return 1
        column_values = node.count_rows(subdataset,node.column)
        for key in column_values:
            dat,clas = self.split_dataset(subdataset,classification,key,node.column)
            tree_node = Node()
            tree_node.question = key
            response = self.tree_learning(tree_node,dat,clas)
            if isinstance(response,dict):
                response = node.plurality(response)
                leaf_node =  Leaf(list(response.keys())[0],list(response.values())[0],tree_node.question,node.column)
                node.branch.append(leaf_node)
            elif response == 1:
                dictionary = node.count_rows(classification,0)
                dictionary = node.plurality(dictionary)
                leaf_node = Leaf(list(dictionary.keys())[0],list(dictionary.values())[0],node.question,node.column)
                node.branch.append(leaf_node)
            else: 
                node.branch.append(tree_node)
        return 0
    
    def split_dataset(self,subdataset,classification,value,column):
        array_return = []
        classification_return = []
        subdataset_copy = copy.deepcopy(subdataset)
        for i in range(len(subdataset_copy)):
            if subdataset_copy[i][column] == value:
                del subdataset_copy[i][column]
                array_return.append(subdataset_copy[i])
                classification_return.append(classification[i])
        return array_return,classification_return

    def prune(self,ratio):
        self.pruning_ratio = ratio
        self.pruning(self.root)

    def pruning(self,node):
        if isinstance(node,Leaf):
            return node,False
        response = True
        for i in range(len(node.branch)):
            leaf,pruned = self.pruning(node.branch[i])
            if not isinstance(leaf,Leaf):
                response = False
            if pruned:
                node.branch[i] = leaf
        if response:
            if node.gain <= self.pruning_ratio:
                new_leaf = Leaf(None,0,node.question,node.column)
                summ = {}
                for i in node.branch:
                    if i.prediction not in summ:
                        summ[i.prediction] = i.sample_size
                    else:
                        summ[i.prediction] += i.sample_size 
                max_predictor = max(summ, key = summ.get)
                new_leaf.prediction = max_predictor
                new_leaf.sample_size = summ[max_predictor]
                return new_leaf,True
        return node,False



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
    ["Examen", 'Seminario', 'S', 'Medio', 'B']
]

DT = DecisionTree()
DT.fit(dataset,classification)

DT.root.print_tree(0)

DT.prune(0.92)
print("------------------------------------------")
DT.root.print_tree(0)