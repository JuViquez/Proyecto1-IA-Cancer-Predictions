from source.models.decisiontree.DecisionTreeNode import DecisionTreeNode
class Leaf(DecisionTreeNode):
    def __init__(self,
                prediction,
                sample_size,
                question,
                column):
        super().__init__(question,column)
        self.prediction = prediction
        self.sample_size = sample_size

    def print_tree(self,num):
        print('  '*num+"  HOJA "+self.question +str(self.prediction))