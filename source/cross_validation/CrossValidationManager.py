import numpy as np

class CrossValidationManager(object):
    
    def __init__(self, learner, X, y, loss_function, k = 10):
        self.learner = learner
        self.k = k
        self.X = X
        self.y = y
        self.err_t = []
        self.err_v = []
        self.loss_function = loss_function
        
    def partition(self, fold):
        if(fold >= self.k):
            raise ValueError("fold must be a number between 0 and k-1 ")
        X_length = len(self.X)
        fold_size = X_length // self.k
        lower_index = fold * fold_size
        upper_index = lower_index  + fold_size
        X_test = self.X[lower_index : upper_index]
        y_test = self.y[lower_index : upper_index]
        
        del_range = list(range(lower_index, upper_index))
        X_train = np.delete(self.X, del_range, axis = 0 )
        y_train = np.delete(self.y, del_range, axis = 0 )
        return X_train, y_train, X_test, y_test
    
    def error_rate(self, X, y):
        num_samples = len(X)
        loss = 0
        for i in range(num_samples):
            hx = self.learner.predict(X[i])
            loss += self.loss_function(y[i], hx)
        emp_loss = loss / num_samples
        return emp_loss
    
    def cross_validation_wrapper(self):
        size = 0
        while True: 
            self.size = size + 1
            err_t, err_v = self.cross_validation()
            self.err_t.insert(size, err_t)
            self.err_v.insert(size, err_v)
            if size != 0 and self.err_v[size] > self.err_v[size-1]:
                best_size = size + 1
                self.learner.size = best_size
                self.learner.fit(self.X, self.y)
                return self.learner
    
    def cross_validation(self):
        fold_err_t = 0
        fold_err_v = 0
        fold = 0
        for fold in range(self.k):
            X_train, y_train, X_test, y_test = self.partition(self, fold)
            self.learner.fit(X_train, y_train)
            fold_err_t +=  self.error_rate(X_train, y_train)
            fold_err_v += self.error_rate(X_test, y_test)
        mean_fold_err_t = fold_err_t / self.k
        mean_fold_err_v = fold_err_v / self.k
        return mean_fold_err_t, mean_fold_err_v
    
    
            
            
    