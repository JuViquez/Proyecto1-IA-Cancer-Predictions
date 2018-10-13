import numpy as np

class CrossValidationManager(object):
    
    def __init__(self, learner, k = 10, X, y, loss_function):
        self.learner = learner
        self.k = k
        self.X = X
        self.y = y
        self.err_t = []
        self.err_v = []
        
    def partition(self, fold):
        if(fold > self.k):
            raise ValueError("fold must be a number between 0 and k ")
        X_length = len(self.X)
        fold_size = X_length // self.k
        lower_index = fold * fold_size
        upper_index = lower_index  + fold_size
        X_test = X[lower_index : upper_index]
        y_test = y[lower_index : upper_index]
        
        del_range = list(range(lower_index, upper_index))
        X_train = np.delete(X, del_range, axis = 0 )
        y_train = np.delete(y, del_range, axis = 0 )
        return X_train, y_train, X_test, y_test
    
    def error_rate(self, X, y):
        num_samples = len(X)
        loss = 0
        for i in range(num_samples):
            hx = self.learner.predict(X[i])
            loss += loss_function(y[i], hx)
        emp_loss = loss / num_samples
        return emp_loss
    
    def cross_validation_wrapper(self):
        size = 0
        while True: 
            err_t, err_v = self.cross_validation(size+1)
            self.err_t.insert(size, err_t)
            self.err_v.insert(size, err_v)
            if self.err_v[-1] > self.err_v[-2]:
                best_size = size - 2
                self.learner.size = best_size
                self.learner.fit(self.X, self.y)
                return self.learner
    
    def cross_validation(self, size):
        fold_err_t = 0
        fold_err_v = 0
        fold = 0
        self.learner.size = size
        for fold in range(self.k):
            X_train, y_train, X_test, y_test = self.partition(self, fold)
            self.learner.fit(X_train, y_train)
            fold_err_t +=  self.error_rate(X_train, y_train)
            fold_err_v += self.error_rate(X_test, y_test)
        mean_fold_err_t = fold_err_t / self.k
        mean_fold_err_v = fold_err_v / self.k
        return mean_fold_err_t, mean_fold_err_v
    
    
            
            
    