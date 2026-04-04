import numpy as np

class Node:
    def __init__(self,best_feature=None,threshold=None,n_samples=None,left=None,node_rss=None,right=None,*,value=None):
        self.best_feature = best_feature
        self.n_samples = n_samples # number of training samples at this node
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 
        self.node_rss = node_rss
    def is_leaf_node(self):
        return self.value is not None



class DecisionTreeRegressor:
    def __init__(self,max_depth=100,min_samples_split=2):
        self.max_depth = max_depth # max splits in this case 2^100 leaf nodes
        self.min_samples_split = min_samples_split # minimum samples per leaf node
    
    def fit(self,X,y):
        self.root = self._build_tree(X,y,depth=0)

    def _build_tree(self,X,y,depth):
        n_samples = X.shape[0]
        node_rss = np.sum((y-np.mean(y))**2)
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split) or (len(np.unique(y)) == 1):
            return Node(value=np.mean(y),node_rss=node_rss,n_samples=n_samples)
        best_feat,best_t = self._best_split(X,y)
        if best_feat is None:
            return Node(value=np.mean(y),node_rss=node_rss,n_samples=n_samples)
        left_mask = X[:,best_feat] <= best_t
        right_mask = X[:,best_feat] > best_t
        X_left,y_left = X[left_mask],y[left_mask]
        X_right,y_right = X[right_mask],y[right_mask]
        left_child = self._build_tree(X_left,y_left,depth+1)
        right_child = self._build_tree(X_right,y_right,depth+1)
        return Node(best_feature=best_feat,threshold=best_t,n_samples=len(y),node_rss=node_rss,left=left_child,right=right_child)
    
    def _best_split(self,X,y):
        best_rss = float("inf")
        best_feat = None
        best_t = None
        n_features = X.shape[1]
        for j in range(n_features):
            column_values = np.unique(X[:,j])
            for t in column_values:
                left_group = X[:,j] <= t
                right_group = X[:,j] > t
                if np.sum(left_group) == 0 or np.sum(right_group) == 0:
                    continue
                #computes rss
                rss_left = np.sum((y[left_group]-y[left_group].mean())**2)
                rss_right = np.sum((y[right_group]-y[right_group].mean())**2)
                total_rss = rss_left+rss_right
                if total_rss < best_rss:
                    best_rss = total_rss
                    best_feat = j
                    best_t = t
        if best_feat is None:
            return None,None
        return best_feat,best_t
    
    def predict(self,X_new):
        return np.array([self._traverse(x) for x in X_new] )
    def _traverse(self,x):
        node = self.root
        while not node.is_leaf_node():
            if x[node.best_feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
            
        




     