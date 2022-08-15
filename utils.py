from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import glob
import os
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer,LabelEncoder,MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from calculate_metric_score import get_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeavePOut,LeaveOneOut
import os 
import random
import numpy as np 
import torch
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import KernelPCA
import pickle

def seedBasic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def seedTorch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
def seedEverything(seed=0):
    seedBasic(seed)
    seedTorch(seed)
    
grid_parameters_name  = ['knn_args','SVC_args','rf_args','lr_args','NB_args']
grid_parameters =[
            {
                'clf': [KNeighborsClassifier()],
                'clf__n_neighbors': [5,7],
                'clf__algorithm': ['auto'],
                # 'clf__weights': ['uniform', 'distance']
            },

            {
                'clf': [SVC()],
                'clf__C': [1, 2, 5],
                'clf__probability': [True]
            },

            {
                'clf': [RandomForestClassifier()],
                'clf__n_estimators': [10,100],
            },
                        {
                'clf': [LogisticRegression()],
                'clf__C': [1, 2, 5],
            },

            {
                'clf': [BernoulliNB()],
                'clf__alpha': [1,2,10,100],
            }
            
        ]

def get_cv_split(x_train):
    n_splits = None
    if len(x_train) < 50:
        skf = LeavePOut(2)
        n_splits = skf.get_n_splits(x_train)
        name = 'Leave-pair-out'
    elif len(x_train) < 100:
        skf = LeaveOneOut()
        n_splits = skf.get_n_splits(x_train)
        name = 'LOOCV (leave-one-out)'
    elif len(x_train) < 1000:
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)
        name = 'Folds CV10'
    else:
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits)
        name = '5 Folds CV'
    return skf,n_splits,name

def get_specific_df(row):
    p = glob.glob('data/microarrays/data/*/*.csv')
    sorted(p)
    return pd.read_csv(p[row]).sample(frac=1),p[row].split("/")[-1]

def get_data(num_row):
    summary_data = []

    for idx, csv in tqdm(enumerate(glob.glob('data/microarrays/data/*/*.csv')[:num_row])):
        # print(csv)
        df = pd.read_csv(csv)
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['target'])

        y_train = train_df.pop('target')
        y_train = y_train.to_numpy()
        X_train = train_df.to_numpy()

        y_test = test_df.pop('target')
        y_test = y_test.to_numpy()
        X_test = test_df.to_numpy()

        # clf = RandomForestClassifier(random_state=42)
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # acc = accuracy_score(y_test, y_pred)

        features_count = X_train.shape[0]
        samples_count = X_train.shape[1]
        num_classes = len(set(y_test))


        # print(str(idx).zfill(2), 'baseline acc', str(acc)[:5], csv, features_count, samples_count, num_classes)
        summary_data.append({
            'path':csv,
            'name':csv.split(os.sep)[-1],
            'corpus':csv.split(os.sep)[-2],
            'samples': samples_count,
            'features': features_count,        
            'num_classes': num_classes,
            # 'baseline_acc': acc,

                            })
    summary_df = pd.DataFrame(summary_data).sample(frac=1)
    return summary_df

def plot_auc(y_score,y_test,n_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(
        fpr[2],
        tpr[2],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
    return fpr,tpr,roc_auc



class ColumnsPreprocessing(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self, columns):
        self.columns = columns
        self.X = None
        self.y = None
        if os.path.exists(f"output/Preprocessing/{self.columns}"):
            self.X = pd.read_csv(f"output/Preprocessing/{self.columns}")
            self.y = self.X['y']
            self.X = self.X.drop('y',axis=1)
            
    def transform(self, X, return_y = True):
        if self.X is None:
            X,y = preprocessing(X)
            self.X = X
            self.y = y
            df2save = self.X.copy()
            df2save['y'] = self.y
            df2save.to_csv(f"output/Preprocessing/{self.columns}",index=False)
        if return_y:
            return self.X,self.y
        return self.X
    
    def fit_transform(self, X, return_y = True):
        if self.X is None:
            X,y = preprocessing(X)
            self.X = X
            self.y = y
            df2save = self.X.copy()
            df2save['y'] = self.y
            df2save.to_csv(f"output/Preprocessing/{self.columns}",index=False)
        if return_y:
            return self.X,self.y
        return self.X
    
    def fit(self, X, return_y = True):
        if self.X is None:
            X,y = preprocessing(X)
            self.X = X
            self.y = y
            df2save = self.X.copy()
            df2save['y'] = self.y
            df2save.to_csv(f"output/Preprocessing/{self.columns}",index=False)
        if return_y:
            return self.X,self.y
        return self.X
    
    def predict(self, X, y = None):
        return self.transform(X, y)
    
    # def fit_transform(self, X, y):
    #     if self.X is None:
    #         X,y = preprocessing(X)
    #         self.X = X
    #         self.y = y
    #         df2save = self.X.copy()
    #         df2save['y'] = self.y
    #         df2save.to_csv(f"output/Preprocessing/{self.columns}",index=False)
    #     return self.X,self.y
    
def preprocessing(df,columns=None):
    ## LabelEncoder
    LE = LabelEncoder()
    LE.fit(df.target)
    df['target'] = LE.transform(df.target)
    y = df['target'].values
    train_features = df.drop('target',axis=1)
    # train_features = train_features.fillna(np.Inf)
    ## drop columns
    if columns is None:
        columns = [col for col in train_features.columns]
        train_features = train_features[columns]
    ## fillna
    for col in columns:
        if len(list(set(df[col])))<2:
            continue
        transformer = SimpleImputer(missing_values=np.nan, strategy='mean')
        vec_len = len(train_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    
    ##VarianceThreshold
    sel = VarianceThreshold(threshold=0)
    sel.fit(train_features) 
    new_col = [x for x in train_features.columns if x in train_features.columns[sel.get_support()]]
    train_features = train_features[new_col]
    columns = train_features.columns
    train_features = train_features.fillna(0)
    ## PowerTransformer
    for col in columns:
        if len(list(set(df[col])))<2:
            continue
        transformer = PowerTransformer()
        vec_len = len(train_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    return train_features,y


class ColumnsTrainKfold(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self,                    
                 knn_args = {"n_neighbors": 5},
                 rf_args = {"n_estimators":100},
                 lr_args = {"C":1e5},
                 SVC_args = {"C":1,"probability":True},
                 NB_args = {"alpha":1},
                 use_smote = False,
                 use_pca = False,):
        self.knn_args = knn_args
        self.rf_args = rf_args
        self.lr_args = lr_args
        self.SVC_args = SVC_args
        self.NB_args = NB_args
        self.use_smote = use_smote
        self.use_pca = use_pca
        
    def fit(self, X, y = None):
        return train_kfold(X,y,self.knn_args,self.rf_args,self.lr_args,self.SVC_args,self.NB_args,self.use_smote,self.use_pca)
    
    def transform(self, X, y):
        return train_kfold(X,y,self.knn_args,self.rf_args,self.lr_args,self.SVC_args,self.NB_args,self.use_smote,self.use_pca)
    
    def fit_transform(self, X, y):
        return train_kfold(X,y,self.knn_args,self.rf_args,self.lr_args,self.SVC_args,self.NB_args,self.use_smote,self.use_pca)
    
    
class TrainOneFold(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self,                    
                 knn_args = {"n_neighbors": 5},
                 rf_args = {"n_estimators":100},
                 lr_args = {"C":1e5},
                 SVC_args = {"C":1,"probability":True},
                 NB_args = {"alpha":1},
                 use_smote = False,
                 use_pca = False,):
        self.knn_args = knn_args
        self.rf_args = rf_args
        self.lr_args = lr_args
        self.SVC_args = SVC_args
        self.NB_args = NB_args
        self.use_smote = use_smote
        self.use_pca = use_pca
        self.knn = KNeighborsClassifier(**knn_args)
        self.rf = RandomForestClassifier(**rf_args)
        self.lr = LogisticRegression(**lr_args)
        self.svc = SVC(**SVC_args)
        self.BN = BernoulliNB(**NB_args)
        self.models = {"knn":self.knn,"rf":self.rf,"lr":self.lr,"SVC":self.svc,"BN":self.BN}
        
    def fit(self, X, y):
        for m in self.models.keys():
            s = time.time()
            self.models[m].fit(X, y)
        return self
                
    def predict_proba(self, X):
        history = {}
        for m in self.models.keys():
            history[m] =  self.models[m].predict_proba(X)
        return history
    
    def predict(self, X):
        history = {}
        for m in self.models.keys():
            history[m] =  self.models[m].predict(X)
        return history
    
    def transform(self, X):
        history = {}
        for m in self.models.keys():
            history[m] =  self.models[m].predict_proba(X)
        return history
        
        
    
    def fit_transform(self, X, y):
        return train_kfold(X,y,self.knn_args,self.rf_args,self.lr_args,self.SVC_args,self.NB_args,self.use_smote,self.use_pca)
    
def pca(train, test,kernel='linear',n_components=7):
    transformer = KernelPCA(n_components=n_components, kernel='linear')
    transformer2 = KernelPCA(n_components=n_components, kernel='rbf')
    data = pd.concat([pd.DataFrame(train), pd.DataFrame(test)])
    data2 = transformer.fit_transform(data)
    data3 = transformer2.fit_transform(data)
    train2 = data2[:train.shape[0]]; test2 = data2[-test.shape[0]:]
    train3 = data3[:train.shape[0]]; test3 = data3[-test.shape[0]:]
    train_features = np.concatenate((train, train2,train3), axis=1)
    test_features = np.concatenate((test, test2,test3), axis=1)
    return train_features, test_features

def train_kfold(X,
                   y,
                   knn_args = {"n_neighbors": 5},
                   rf_args = {"n_estimators":100},
                   lr_args = {"C":1e5},
                   SVC_args = {"C":1,"probability":True},
                   NB_args = {"alpha":1},
                   use_smote = False,
                   use_pca = False,
                  ):
    knn = KNeighborsClassifier(**knn_args)
    rf = RandomForestClassifier(**rf_args)
    lr = LogisticRegression(**lr_args)
    svc = SVC(**SVC_args)
    BN = BernoulliNB(**NB_args)
    models = {"knn":knn,"rf":rf,"lr":lr,"SVC":svc,"BN":BN}
    history = {}
    skf,n_splits,name_cv = get_cv_split(X)
    for m in models.keys():
        s = time.time()
        history[m] = {}
        history[m]['y_score'] = []
        history[m]['y_pred'] = []
        history[m]['y_val'] = []
        history[m]['index_val'] = []
        split_fun = skf.split(X, y) if "Folds" in name_cv else skf.split(X)
        for fold,(train_index, test_index) in enumerate(split_fun):
            X_train, X_test = X[train_index].copy(), X[test_index].copy()
            y_train, y_test = y[train_index].copy(), y[test_index].copy()
            if use_pca:
                # X_train,X_test = pca(X_train,X_test,kernel='linear',n_components=min(7,len(X_train)//2))
                X_train,X_test = pca(X_train,X_test,kernel='rbf',n_components=min(7,len(X_train)//2))
            if use_smote:
                sm = SMOTE(random_state=0)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            models[m].fit(X_train, y_train)
            y_score = models[m].predict_proba(X_test)
            y_pred = models[m].predict(X_test)
            history[m]['y_score'].append(y_score)
            history[m]['y_pred'].append(y_pred)
            history[m]['y_val'].append(y_test)
            history[m]['index_val'].append(test_index)
        history[m]['y_score'] = np.concatenate(history[m]['y_score'])
        history[m]['y_pred'] = np.concatenate(history[m]['y_pred'])
        history[m]['y_val'] = np.concatenate(history[m]['y_val'])
        history[m]['index_val'] = np.concatenate(history[m]['index_val'])
        history[m]['score'] = get_score(history[m]['y_val'],history[m]['y_score'])
        history[m]['train_time'] = time.time()-s
        history[m]['n_splits'] = n_splits
        history[m]['name_cv'] = name_cv
        infer_time = time.time()
        _ = models[m].predict_proba(X_train)
        history[m]['infer_time'] =  time.time() - infer_time
    return history

def train_one_fold(X,
                   y,
                   knn_args = {"n_neighbors": 5},
                   rf_args = {"n_estimators":100},
                   lr_args = {"C":1e5},
                   SVC_args = {"C":1,"probability":True},
                   NB_args = {"alpha":1},
                   use_smote = False,
                   use_pca = False,
                  ):
    knn = KNeighborsClassifier(**knn_args)
    rf = RandomForestClassifier(**rf_args)
    lr = LogisticRegression(**lr_args)
    svc = SVC(**SVC_args)
    BN = BernoulliNB(**NB_args)
    models = {"knn":knn,"rf":rf,"lr":lr,"SVC":svc,"BN":BN}
    history = {}
    skf,n_splits,name_cv = get_cv_split(X)
    for m in models.keys():
        s = time.time()
        history[m] = {}
        history[m]['y_score'] = []
        history[m]['y_pred'] = []
        history[m]['y_val'] = []
        history[m]['index_val'] = []
        split_fun = skf.split(X, y) if "Folds" in name_cv else skf.split(X)
        for fold,(train_index, test_index) in enumerate(split_fun):
            X_train, X_test = X[train_index].copy(), X[test_index].copy()
            y_train, y_test = y[train_index].copy(), y[test_index].copy()
            if use_pca:
                # X_train,X_test = pca(X_train,X_test,kernel='linear',n_components=min(7,len(X_train)//2))
                X_train,X_test = pca(X_train,X_test,kernel='rbf',n_components=min(7,len(X_train)//2))
            if use_smote:
                sm = SMOTE(random_state=0)
                X_train, y_train = sm.fit_resample(X_train, y_train)

            models[m].fit(X_train, y_train)
            y_score = models[m].predict_proba(X_test)
            y_pred = models[m].predict(X_test)
            history[m]['y_score'].append(y_score)
            history[m]['y_pred'].append(y_pred)
            history[m]['y_val'].append(y_test)
            history[m]['index_val'].append(test_index)
        history[m]['y_score'] = np.concatenate(history[m]['y_score'])
        history[m]['y_pred'] = np.concatenate(history[m]['y_pred'])
        history[m]['y_val'] = np.concatenate(history[m]['y_val'])
        history[m]['index_val'] = np.concatenate(history[m]['index_val'])
        history[m]['score'] = get_score(history[m]['y_val'],history[m]['y_score'])
        history[m]['train_time'] = time.time()-s
        history[m]['n_splits'] = n_splits
        history[m]['name_cv'] = name_cv
        infer_time = time.time()
        _ = models[m].predict_proba(X_train)
        history[m]['infer_time'] =  time.time() - infer_time
    return history

def pkl2csv(Filtering_Algorithm,
            datsets_num,
            path = 'temp2',
            name = None,
            topk=[1,2,3,4,5,10,15,20,25,30,50,100]
           ):
    try:
        if name is None:
            df,name = get_specific_df(datsets_num)
        history_df = []
        with open(f'{path}/{Filtering_Algorithm}/{name}_history.pickle', 'rb') as handle:
             results = pickle.load(handle)
        history_df = []
        if len(results) == 1:
            topk = [1]
        for i,k in enumerate(topk):
            history = results[i]
            for key in history.keys():
                if key.lower() in ["knn","rf","lr","svc","bn"]:
                    break
            if len(topk) == 1:
                k = history[key]['k']
            name_cv = history[key]['name_cv']
            n_splits = history[key]['n_splits']
            score_gates = history['score_gates']
            col_name = history['col_name']
            fs_time = history['time_gates']
            kfold = n_splits < 11        
            for m in ['knn', 'rf', 'lr', 'SVC', 'BN']:
                try:
                    Learning_algorithm_train_time = sum([history[m][fold]['train_time'] for fold in range(n_splits)])
                    infer_time_time = sum([history[m][fold]['infer_time'] for fold in range(n_splits)])
                    for metric in ['ACC','MCC','AUC','PR-AUC']:

                        history_df.append({
                                    "Dataset Name": name,
                                    "Number of samples":len(df),
                                    "Original Number of features":df.shape[1],
                                    "Filtering Algorithm":Filtering_Algorithm,
                                    "Learning algorithm":m,
                                    "Number of features selected (K)":k,
                                    "CV Method": name_cv,
                                    "Fold":"CV Score",
                                    "CV folds split":n_splits,
                                    "List of Selected Features Names (Long STRING)":col_name[:k],
                                    "Selected Features scores":score_gates[:k],
                                    "Measure Type":metric,
                                    "Measure Value":history[m]['score']['cv_score'][metric],
                                    "Learning algorithm train time":Learning_algorithm_train_time,
                                    "Learning algorithm predict proba time":infer_time_time,
                                    "Filtering algorithm time":fs_time,
                                                              })
                except:
                    pass
                

            if kfold:
                score_gates = [history[f'fold_{f}_score_gates'] for f in range(n_splits)]
                col_name = [history[f'fold_{f}_col_name'] for f in range(n_splits)]
                fs_time = [history[f'fold_{f}_fs_time'] for f in range(n_splits)]
                for m in ['knn', 'rf', 'lr', 'SVC', 'BN']:
                    try:
                        for fold in range(n_splits):
                            for metric in ['ACC','MCC','AUC','PR-AUC']:
                                history_df.append({
                                            "Dataset Name": name,
                                            "Number of samples":len(df),
                                            "Original Number of features":df.shape[1],
                                            "Filtering Algorithm":Filtering_Algorithm,
                                            "Learning algorithm":m,
                                            "Number of features selected (K)":k,
                                            "CV Method": name_cv,
                                            "Fold":fold,
                                            "CV folds split":n_splits,
                                            "List of Selected Features Names (Long STRING)":col_name[fold][:k],
                                            "Selected Features scores":score_gates[fold][:k],
                                            "Measure Type":metric,
                                            "Measure Value":history[m][fold]['score'][metric],
                                            "Learning algorithm train time":history[m][fold]['train_time'],
                                            "Learning algorithm predict proba time":history[m][fold]['infer_time'],
                                            "Filtering algorithm time":fs_time[fold],
                                                              })
                    except:
                        pass


        return pd.DataFrame(history_df)
    except:
        return pd.DataFrame()
