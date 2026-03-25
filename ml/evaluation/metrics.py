import numpy as np

def r2_score(y_true,y_pred):
    num = np.sum((y_pred-y_true)**2) #model error
    denom = np.sum((y_true-np.mean(y_true))**2) #baseline error
    return 1- (num/denom)
    
def mean_squared_error(y_true,y_pred):
    return np.mean((y_pred-y_true)**2)
    
def mean_absolute_error(y_true,y_pred):
    return np.mean(abs(y_pred-y_true))
    
def root_mean_squared_error(y_true,y_pred):
    return np.sqrt(np.mean((y_pred-y_true)**2))

def accuracy(y_true,y_pred):
    return (y_pred == np.array(y_true)).mean()




