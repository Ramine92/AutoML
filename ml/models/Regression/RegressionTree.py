import numpy as np
import copy
from ml.evaluation.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
class Node:
    def __init__(self,best_feature=None,threshold=None,n_samples=None,left=None,node_rss=None,right=None,mean_value=None,*,value=None):
        self.best_feature = best_feature
        self.n_samples = n_samples # number of training samples at this node
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 
        self.node_rss = node_rss
        self.mean_value = mean_value # np.mean(y)
    def is_leaf_node(self):
        return self.value is not None



class DecisionTreeRegressor:
    def __init__(self,max_depth=10,min_samples_split=10):
        self.max_depth = max_depth # max splits in this case 2^10 leaf nodes
        self.min_samples_split = min_samples_split # minimum samples per leaf node
    
    def fit(self,X,y,prune=False):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X,y,depth=0)
        if prune:
            alpha = self.find_best_alpha(X,y)
            self._prune_with_alpha(self.root,alpha)

    def _build_tree(self,X,y,depth):
        n_samples = X.shape[0]
        node_rss = np.sum((y-np.mean(y))**2)
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split) or (len(np.unique(y)) == 1):
            return Node(value=np.mean(y),mean_value=np.mean(y),node_rss=node_rss,n_samples=n_samples)
        best_feat,best_t = self._best_split(X,y)
        if best_feat is None:
            return Node(value=np.mean(y),mean_value=np.mean(y),node_rss=node_rss,n_samples=n_samples)
        left_mask = X[:,best_feat] <= best_t
        right_mask = X[:,best_feat] > best_t
        X_left,y_left = X[left_mask],y[left_mask]
        X_right,y_right = X[right_mask],y[right_mask]
        left_child = self._build_tree(X_left,y_left,depth+1)
        right_child = self._build_tree(X_right,y_right,depth+1)
        return Node(best_feature=best_feat,threshold=best_t,n_samples=len(y),mean_value=np.mean(y),node_rss=node_rss,left=left_child,right=right_child)
    
    def _best_split(self,X,y):
        best_rss = float("inf")
        best_feat = None
        best_t = None
        n_features = X.shape[1]
        for j in range(n_features):
            column_values = np.unique(X[:,j])
            if len(column_values) > 50:
                column_values = np.random.choice(column_values,50,replace=False)
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
        X_new_arr = np.array(X_new, dtype=float)
        return np.array([self._traverse(x) for x in X_new_arr] )
    def _traverse(self,x):
        node = self.root
        while not node.is_leaf_node():
            if x[node.best_feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _leaf_count(self,node):
        if node.is_leaf_node():
            return 1
        return self._leaf_count(node.left) + self._leaf_count(node.right)
    
    def _subtree_rss(self,node):
        if node.is_leaf_node():
            return node.node_rss
        return self._subtree_rss(node.left)+self._subtree_rss(node.right)
    

    def _effective_alpha(self,node):
        alpha_effective = (node.node_rss-self._subtree_rss(node) ) / (self._leaf_count(node) -1)
        return alpha_effective
    
    def _find_weakest_link(self,node):
        if node.is_leaf_node():
            return (float("inf"),None)
        my_alpha = self._effective_alpha(node)
        left_alpha,left_node = self._find_weakest_link(node.left)
        right_alpha,right_node = self._find_weakest_link(node.right)
        if my_alpha <= left_alpha and my_alpha <= right_alpha:
            return (my_alpha,node)        
        elif left_alpha <= right_alpha:
            return (left_alpha,left_node)
        else:
            return (right_alpha,right_node)

    def ccp_alphas(self):
        alphas = [0.0]
        root = copy.deepcopy(self.root)
        while not root.is_leaf_node():
            #1 Find Weakest Link
            min_alpha,weakest_node = self._find_weakest_link(root)

            #2 PRUNE: turn weakest node into a leaf
            weakest_node.value = weakest_node.mean_value
            weakest_node.left = None
            weakest_node.right = None

            #3. Save this alpha
            alphas.append(min_alpha)
        return alphas
        
    def _prune_with_alpha(self,node,alpha):
        if node.is_leaf_node():
            return
        if self._effective_alpha(node) <= alpha:
            node.value = node.mean_value
            node.left = None
            node.right = None
            return
        self._prune_with_alpha(node.left,alpha)
        self._prune_with_alpha(node.right,alpha)
    
    def find_best_alpha(self, X, y, k_folds=5):
        # 1. Get candidate alphas from the FULL tree
        alphas = self.ccp_alphas()
        
        # 2. Create folds
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        folds = np.array_split(indices, k_folds)
        
        # 3. Test each alpha
        best_alpha = 0
        best_mse = float('inf')
        
        for alpha in alphas:
            fold_mses = []
            
            for j in range(k_folds):
                # Split into train/val
                val_idx = folds[j]
                train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != j])
                
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                # Train a FRESH tree
                temp_tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split)
                temp_tree.fit(X_train, y_train)
                
                # Prune it with this alpha
                temp_tree._prune_with_alpha(temp_tree.root, alpha)
                
                # Score on validation
                preds = temp_tree.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                fold_mses.append(mse)
            
            avg_mse = np.mean(fold_mses)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha = alpha
        
        return best_alpha
    
    def score(self,X_new,y_true):
        predictions = self.predict(X_new)
        mse = mean_squared_error(y_true,predictions)
        mae = mean_absolute_error(y_true,predictions)
        rmse = root_mean_squared_error(y_true,predictions)
        r2 = r2_score(y_true,predictions)
        return {"MSE":mse,"MAE":mae,"RMSE":rmse,"R2":r2}





            
        




     