import numpy as np

class LogisticRegression:

    def __init__(self,iterations=1000,alpha=0.01):
        self.alpha = alpha
        self.iterations = iterations

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    def predict_proba(self,X_new):
        if not hasattr(self,'theta'):
            raise Exception("this instance is not fitted yet. Call 'fit' first.")

        m_new = X_new.shape[0] # rows
        X_new_with_bias = np.c_[(np.ones(shape=(m_new,1))),X_new]
        predictions = np.dot(X_new_with_bias,self.theta)
        return self.sigmoid(predictions)

    def predict(self,X_new):
        proba = self.predict_proba(X_new) # (m,1)
        return (proba >= 0.5).astype(int) # 0 or 1
    
    def estimate_error(self,y,predictions):
        return predictions - y

    def gradient(self,X,error):
        return (1/self.m)*np.dot((X).T,error)
    def update_weights(self,gradient):
        self.theta = self.theta -(self.alpha*gradient)
        return self.theta
    
    def fit(self,X,y):
        self.m = X.shape[0] # rows
        self.n = X.shape[1] #columns
        self.y = np.ravel(y)
        self.X = np.c_[(np.ones(shape=(self.m,1))),X]
        self.theta = np.random.randn(self.n+1,1)
        for _ in range(self.iterations):
            predictions = np.dot(self.X,self.theta)
            proba = self.sigmoid(predictions)
            error = self.estimate_error(self.y,proba)
            gradient = self.gradient(self.X,error)
            self.update_weights(gradient)





    
    



