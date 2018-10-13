from source.decisiontree.Node import Node
from source.decisiontree.Leaf import Leaf

class DecisionTree():
    def __init__(self):
        self.root = None
    
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
            dat,clas = node.split_dataset(subdataset,classification,key)
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

DT = DecisionTree()
DT.fit(dataset,classification)

DT.root.print_tree(0)