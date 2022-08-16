import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from skfeature.function.similarity_based import reliefF
import reliefF_article
import reliefF_adj

"""
This class is an implementation of the Feature selection suggested algorithm from the article:

HongFang Zhou, JiaWei Zhang, YueQing Zhou, XiaoJie Guo, YiMing Ma, 2022. A feature selection algorithm of decision trees based on feature weight.

https://www.sciencedirect.com/science/article/pii/S0957417420306515
Methods
—---------
conti_to_discrete(self,in_col_data) 
	Using the K-means algorithm, divide a continuous feature into 2-10 bins.
discrete_all_features (in_data) 
	Using the conti_to_discrete method, discretize all features

features_filter_using_relief(X,y,ret_col_index=False,use_improvment=False)
	Using the adjusted releaseF algorithm with dataset and label as input, and two additional flags: ret_col_index which returns a column index or dataset with selected features, and use_improvment which uses the original implementation from the article or our improvement.

fit_transform(X,y,use_improvment=False)
	For compatibility with Scikit-Learn

"""
class FWDT():
    
    def __init__(self,max_discerete_features=10):
        self.max_discerete_features=max_discerete_features,
    
    def is_continues(self,in_data,in_max_uniques=10):
        if len(np.unique(in_data))>in_max_uniques:
            return True
        return False

    def conti_to_discrete(self,in_col_data):
        best_k = -1
        max_dist_calc = 0
        best_pred = None
        rnd_index = np.random.choice(len(in_col_data), size=1, replace=False)
        for k in np.arange(2,11):

            km = KMeans(n_clusters=k)
            km.fit(in_col_data.reshape(-1, 1))
            preds = km.predict(in_col_data.reshape(-1, 1))
            #rnd_index = np.random.choice(len(in_col_data), size=1, replace=False)
            rnd_index_val = in_col_data[rnd_index]
            rnd_index_cluster = preds[rnd_index][0]
            cluster_members = in_col_data[np.where(preds == rnd_index_cluster)]
            non_cluster_members = in_col_data[np.where(preds != rnd_index_cluster)]
            
            cluster_dist = distance.euclidean(cluster_members.reshape(-1, 1), rnd_index_val)
            non_cluster_dist = distance.euclidean(non_cluster_members.reshape(-1, 1), rnd_index_val)

            ret_val = (non_cluster_dist - cluster_dist)/max(non_cluster_dist , cluster_dist)
            if ret_val > max_dist_calc:
                best_k = k
                max_dist_calc = ret_val
                best_pred = preds
        return best_k,max_dist_calc,best_pred

    def discrete_all_features(self,in_data):
        for col_i in np.arange(0,in_data.shape[1]):
            cur_data = in_data[:,col_i]
            if (self.is_continues(cur_data)):
                _,_,in_data[:,col_i] = self.conti_to_discrete(cur_data)

        return in_data

    def features_filter_using_relief(self, X,y,ret_col_index=False,use_improvment=False):
        num_of_features = X.shape[1]
        # threshold = num_of_features/2-1
        if use_improvment:
            ret_features_index = reliefF_adj.reliefF(X,y,mode='raw')
        else:
            ret_features_index = reliefF_article.reliefF(X,y,mode='raw')
        
        ret_features_index = np.array(ret_features_index)
        if (use_improvment):
            threshold = np.median(ret_features_index)
        else:
            threshold = np.median(ret_features_index)
        
        if(ret_col_index):
            ret_features_index[ret_features_index<threshold]=-999999
            return ret_features_index
        else:
            return X[:, ret_features_index[ret_features_index<threshold]]
    
    def fit_transform(self,X,y,use_improvment=False):
        new_X = self.discrete_all_features(X)
        ret = self.features_filter_using_relief(new_X,y,ret_col_index=True,use_improvment=use_improvment)
        return ret
    
    def transform(X,y):
        new_X = self.discrete_all_features(X)
        
    def fit_transform_df(self,X,y,use_improvment=False):
        col_list = X.columns.values
        new_X = self.discrete_all_features(X.values)
        ret = self.features_filter_using_relief(new_X,y.values,ret_col_index=True,use_improvment=use_improvment)
        return ret
        
