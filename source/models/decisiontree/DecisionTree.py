import copy
from source.models.Model import Model
from source.models.decisiontree.Node import Node
from source.models.decisiontree.Leaf import Leaf


class DecisionTree(Model):
    def __init__(self):
        super().__init__(0)
        self.root = None
        self.pruning_ratio = 0

    def fit(self, X, Y):
        self.root = Node()
        self.output = ""
        self.tree_learning(self.root, X, Y, 0)

    def tree_learning(self, node, subdataset, classification, out_level):
        out_level += 1
        if(len(subdataset) == 0):
            return 1
        classification_values = node.count_rows(classification, 0)
        if (len(classification_values) < 2 or len(subdataset[0]) == 0):
            return classification_values
        node.best_gain(subdataset, classification)
        if(node.gain == 0):
            return 1
        column_values = node.count_rows(subdataset, node.column)
        for key in column_values:
            dat, clas = self.split_dataset(
                subdataset, classification, key, node.column)
            tree_node = Node()
            tree_node.question = key
            response = self.tree_learning(tree_node, dat, clas, out_level)
            if isinstance(response, dict):
                response = node.plurality(response)
                leaf_node = Leaf(
                    list(
                        response.keys())[0], list(
                        response.values())[0], tree_node.question, node.column)
                node.branch.append(leaf_node)
            elif response == 1:
                dictionary = node.count_rows(classification, 0)
                dictionary = node.plurality(dictionary)
                leaf_node = Leaf(
                    list(
                        dictionary.keys())[0], list(
                        dictionary.values())[0], node.question, node.column)
                node.branch.append(leaf_node)
            else:
                node.branch.append(tree_node)
        return 0

    def split_dataset(self, subdataset, classification, value, column):
        array_return = []
        classification_return = []
        subdataset_copy = copy.deepcopy(subdataset)
        for i in range(len(subdataset_copy)):
            if subdataset_copy[i][column] == value:
                del subdataset_copy[i][column]
                array_return.append(subdataset_copy[i])
                classification_return.append(classification[i])
        return array_return, classification_return

    def prune(self, ratio):
        self.pruning_ratio = ratio
        if ratio > 0:
            self.pruning(self.root)

    def pruning(self, node):
        if isinstance(node, Leaf):
            return node, False
        response = True
        for i in range(len(node.branch)):
            leaf, pruned = self.pruning(node.branch[i])
            if not isinstance(leaf, Leaf):
                response = False
            if pruned:
                node.branch[i] = leaf
        if response:
            if node.gain <= self.pruning_ratio:
                new_leaf = Leaf(Node, 0, node.question, node.column)
                summ = {}
                for i in node.branch:
                    if i.prediction not in summ:
                        summ[i.prediction] = i.sample_size
                    else:
                        summ[i.prediction] += i.sample_size
                max_predictor = max(summ, key=summ.get)
                new_leaf.prediction = max_predictor
                new_leaf.sample_size = summ[max_predictor]
                return new_leaf, True
        return node, False

    def predict(self, X):
        row = copy.deepcopy(X)
        return self.prediction(self.root, row)

    def prediction(self, node, row):
        for branch in node.branch:
            if branch.question == row[node.column]:
                del row[node.column]
                if isinstance(branch, Leaf):
                    return branch.prediction
                return self.prediction(branch, row)
        return None

    def print_tree(self, node, out_level):
        out_level += 1
        if isinstance(node, Node):
            self.output += "   " * out_level + " Node(gain:" + str(node.gain) + " value:" + str(
                node.question) + " column:" + str(node.column) + ") \n"
            for branch in node.branch:
                self.print_tree(branch, out_level)
        else:
            self.output += "   " * out_level + " Leaf(prediction:" + str(
                node.prediction) + " value:" + str(node.question) + " column:" + str(node.column) + ") \n"
