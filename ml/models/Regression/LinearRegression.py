import numpy as np

class LinearRegression:

    def __init__(self,alpha=0.01,iterations=1000):
        self.alpha = alpha
        self.iterations = iterations

    def predict(self,X_new):
        if not hasattr(self,'theta'):
            raise Exception("this instance is not fitted yet. Call 'fit' first.")
        xnew_m = X_new.shape[0]
        X_new_arr = np.array(X_new, dtype=float)   # ← convert to clean numpy array!
        X_new = np.c_[np.ones((xnew_m, 1)), X_new_arr]
        return np.dot(X_new, self.theta)

    def estimate_error(self,predictions):
        return predictions - self.y
    
    def perform_gradient(self,error):
        gradient = (1/self.m) * np.dot((self.X).T,error)
        return gradient
    
    def update_parameters(self,gradient):
        self.theta = self.theta - (self.alpha*gradient)
        return self.theta
    
    def cost(self,predictions):
        return (1/(2*self.m) ) * np.sum((predictions-self.y)**2)
    
    def fit(self,X,y):
        self.m = X.shape[0] # rows
        self.n = X.shape[1] # columns
        X_arr = np.array(X, dtype=float)
        self.X = np.c_[np.ones((self.m, 1)), X_arr]
        self.y = np.ravel(y)
        self.theta = np.random.randn(self.n+1,1)
        for _ in range(self.iterations):
            prediction = np.dot(self.X,self.theta)
            error = self.estimate_error(prediction) 
            gradient = self.perform_gradient(error)
            self.theta = self.update_parameters(gradient)
        return self.theta
    
    def score(self,X_new,y_true):
        predictions = self.predict(X_new)
        mse = np.mean((predictions-y_true)**2)
        return np.sqrt(mse)
    
    

    
    