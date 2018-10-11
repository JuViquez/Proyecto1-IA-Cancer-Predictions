from DecisionTreeNode import DecisionTreeNode
class Leaf(DecisionTreeNode):
    def __init__(self,
                prediction,
                question,
                column):
        super().__init__(question,column)
        self.prediction = prediction

    def predict(self,row):
        print("Predicción: ")
        return self.prediction

    def print_tree(self,num):
        print('  '*num+"  HOJA "+self.question +str(self.prediction))