from stg import STG
import numpy as np
import scipy.stats # for creating a simple dataset 
import matplotlib.pyplot as plt 
import torch
from sklearn import preprocessing
from sklearn.model_selection import KFold
from utils import get_data,plot_auc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import label_binarize
import time
from sklearn.model_selection import train_test_split
import os
import time
from sklearn.feature_selection import SelectFdr,f_classif
from sklearn.svm import SVC

from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF
from sklearn.feature_selection import RFE
from ensemble_of_filters import ensemble_of_filters, ensemble_of_filters_new
from sklearn.preprocessing import minmax_scale
import sys
sys.path.append('FWDT')
###import FWDT
from FWDT import FWDT


class get_ensemble_class:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = ensemble_of_filters(x,y)
            print(self.gates.shape)
            np.save(f"{self.out_path}/{self.datasets}_ensemble.npy",self.gates)
        return self.gates
    
    
class get_ensemble_class_new:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = ensemble_of_filters_new(x,y)
            print(self.gates.shape)            
            np.save(f"{self.out_path}/{self.datasets}_ensemble.npy",self.gates)
        return self.gates

def FWDT_score_new(X,y):
    cur_f = FWDT()
    ret = cur_f.fit_transform(X,y,use_improvment=True)
    return ret

class get_FWDT_class_new:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = FWDT_score(x,y)
            print(self.gates.shape)
            np.save(f"{self.out_path}/{self.datasets}_FWDT_new.npy",self.gates)
        return self.gates    
    
def FWDT_score(X,y):
    cur_f = FWDT()
    ret = cur_f.fit_transform(X,y)
    return ret

class get_FWDT_class:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = FWDT_score(x,y)
            print(self.gates.shape)
            np.save(f"{self.out_path}/{self.datasets}_FWDT.npy",self.gates)
        return self.gates

def mrmr_score(X,y,k=None):
    selected_features_list = MRMR.mrmr(X,y,mode='index',n_selected_features=X.shape[1])
    ret = -1*np.argsort(selected_features_list) + len(selected_features_list)-1
    return minmax_scale(ret)

class get_mrmr_class:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = mrmr_score(x,y)
            np.save(f"{self.out_path}/{self.datasets}_mrmr.npy",self.gates)
        return self.gates

def reliefF_score(X,y,k=None):
    ret = reliefF.reliefF(X,y,mode='raw')
    return minmax_scale(ret)

class get_reliefF_class:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = reliefF_score(x,y)
            np.save(f"{self.out_path}/{self.datasets}_reliefF.npy",self.gates)
        return self.gates

def RFE_SVM_score_(X,y,k):
    selector = RFE(SVC(kernel='linear'),n_features_to_select=k, step=1)
    ret = selector.fit(X, y)
    return ret.ranking_

def RFE_SVM_score(X,y):
    
    f_arr = np.zeros(X.shape[1])
    #for k in np.arange(1,data_df.values.shape[1]):
    for k in [1,2,3,4,5,10,15,20,25,30,50,100]:
        arr = np.array(RFE_SVM_score_(X,y,k=k))
        f_arr += arr
    ret = -1*np.argsort(f_arr) + len(f_arr)-1
    return minmax_scale(ret)

class get_RFE_SVM_class:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = RFE_SVM_score(x,y)
            np.save(f"{self.out_path}/{self.datasets}_RFE_SVM.npy",self.gates)
        return self.gates

class get_SelectFdr_class:
    def __init__(self,datasets,out_path,alpha=0.1):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        self.alpha=alpha
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = SelectFdr(f_classif,alpha=self.alpha).score_func(x,y)[0]
            self.gates = np.nan_to_num(self.gates,0)/100.0
            np.save(f"{self.out_path}/{self.datasets}_SelectFdr.npy",self.gates)
        return self.gates



class STGConfig:
    task_type='classification'
    hidden_dims=[60, 20]
    activation='tanh'
    optimizer='Adam'
    learning_rate=0.02
    feature_selection = True
    sigma=0.5
    lam=0.5
    random_state=1
    device= "cpu" #"cpu"
    batch_size = 128
    nr_epochs = 2500
    print_interval=5000

class get_stg_class:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = get_stg_gates(x,y)
            np.save(f"{self.out_path}/{self.datasets}_stg.npy",self.gates)
        return self.gates
    
class get_stg_class_new:
    def __init__(self,datasets,out_path):
        self.gates = None
        self.datasets = datasets
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
    def __call__(self,x,y,k=None):
        if self.gates is None:
            self.gates = get_stg_gates(x,y,new=True)
            np.save(f"{self.out_path}/{self.datasets}_stg.npy",self.gates)
        return self.gates
        
    
def get_stg_gates(df,y,new=False):
    s = time.time()
    X_train, X_valid , y_train, y_valid = train_test_split(df,y , test_size=0.2, stratify=y)
    args_cuda = torch.cuda.is_available()
    model = STG(task_type=STGConfig.task_type,
                    input_dim=X_train.shape[1], 
                    output_dim=len(list(set(y_train))), 
                    hidden_dims=STGConfig.hidden_dims, 
                    activation=STGConfig.activation,
                    optimizer=STGConfig.optimizer, 
                    learning_rate=STGConfig.learning_rate, 
                    batch_size=STGConfig.batch_size, 
                    feature_selection=STGConfig.feature_selection, 
                    sigma=STGConfig.sigma,
                    lam=STGConfig.lam, 
                    random_state=STGConfig.random_state, 
                    device=STGConfig.device,
                    CancelOut = new,
                    DotProduct= new,
                    soft_sigmoid= new) 
    model.fit(X_train, 
                  y_train, 
                  nr_epochs=STGConfig.nr_epochs, 
                  valid_X=X_valid, 
                  valid_y=y_valid, 
                  print_interval=STGConfig.print_interval)
    gates = model.get_gates(mode='prob')
    gates_raw = model.get_gates(mode='raw') 
    topk = gates.argsort()
    return np.array(gates)#topk,np.array(df.columns)[topk],gates[topk],time.time()-s