import numpy as np

#regression metrics 

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

#classification metrics

def confusion_counts(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred==1))
    TN = np.sum((y_true == 0) & (y_pred==0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP,TN,FP,FN


def accuracy(y_true,y_pred):
    TP,TN,FP,FN = confusion_counts(y_true,y_pred) # among all classifications which one were correct
    return (TP+TN)/(TP+TN+FP+FN)

def precision(y_true,y_pred):
    TP,TN,FP,FN = confusion_counts(y_true,y_pred)
    return ( TP / (TP+FP) if (TP+FP) > 0 else 0.0) # among all positive predictions how often we were right ?

def recall(y_true,y_pred):
    TP,TN,FP,FN = confusion_counts(y_true,y_pred)
    return TP/(TP+FN) if (TP+FN) > 0 else 0.0 # among everything that was actually positive , did i find it ?

def f1_score(y_true,y_pred):
    return 2* (recall(y_true,y_pred)*precision(y_true,y_pred)) / (recall(y_true,y_pred)+precision(y_true,y_pred))
    #f1 is simply harmonic mean of precision and recall 



