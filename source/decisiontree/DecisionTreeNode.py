class DecisionTreeNode():
    def __init__(self,
                question,
                column):
        self.question = question
        self.column = column
    
    def predict(self,row):
        pass