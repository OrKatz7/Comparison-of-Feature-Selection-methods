print("start")
import os
import sys
from utils import get_data,ColumnsPreprocessing,get_specific_df,ColumnsTrainKfold,seedEverything,grid_parameters_name,grid_parameters,TrainOneFold,get_cv_split
from sklearn.model_selection import GridSearchCV
from stg_fs import get_stg_class,get_SelectFdr_class,get_mrmr_class,get_reliefF_class,get_RFE_SVM_class,get_FWDT_class,get_ensemble_class, get_ensemble_class_new ,get_stg_class_new,get_FWDT_class_new
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import clear_output
from joblib import Parallel, delayed
import pickle
from calculate_metric_score import get_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
# from main import *
from tqdm.auto import tqdm
from IPython.display import clear_output
import torch

def run_GridSearchCV(train_features,y,fun,name,cv=3):
    if os.path.exists(f'output/Preprocessing/{name}.pickle'):
        with open(f'output/Preprocessing/{name}.pickle', 'rb') as handle:
            results = pickle.load(handle)
        return results
    results = {}
    for n,params in tqdm(zip(grid_parameters_name,grid_parameters)):
        clf = params['clf'][0]
        params2 = params.copy()
        params2.pop('clf')
        estimators = [("SelectKBest",SelectKBest(fun,k=100)), ('clf', clf)]
        pipe = Pipeline(estimators)
        grid = GridSearchCV(pipe, param_grid=params2, cv=cv,n_jobs=-1)
        _=grid.fit(train_features,y)
        results[n] = {}
        for row in grid.best_params_:
            results[n][row.split("__")[-1]] = grid.best_params_[row]
    with open(f'output/Preprocessing/{name}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=4)
    return results

def train_kfold(get_stg_fun,
                            Filtering_Algorithm,
                            k,
                            datsets_num,
                            f_outpot,
                            PP,
                            knn_args = {"n_neighbors": 5},
                            rf_args = {"n_estimators":100},
                            lr_args = {"C":1e5},
                            SVC_args = {"C":1,"probability":True},
                            NB_args = {"alpha":1}):

    knn = KNeighborsClassifier(**knn_args)
    rf = RandomForestClassifier(**rf_args)
    lr = LogisticRegression(**lr_args)
    svc = SVC(**SVC_args)
    BN = BernoulliNB(**NB_args)
    models = {"knn":knn,"rf":rf,"lr":lr,"SVC":svc,"BN":BN}
    history = {}
    df,name = get_specific_df(datsets_num)
    skf,n_splits,name_cv = get_cv_split(df)
    train_features,y = PP.transform(df)
    start = time.time()
    score_gates = get_stg_fun(train_features.values,y.values,k=k)
    topk = score_gates.argsort()[::-1][0:k]
    col_name = np.array(df.drop('target',axis=1).columns)[topk]
    score_gates = score_gates[topk]
    history[f"time_gates"] = time.time() - start 
    history[f"score_gates"] = score_gates
    history[f"col_name"] = col_name
    
    score_dict = {}
    run_only_cv = False if "Folds" in name_cv else True
    split_fun = skf.split(train_features, y) if "Folds" in name_cv else skf.split(train_features)
    train_features = train_features.values
    y = y.values
    for clf in models:
        history[clf] = {}
        history[clf]['score'] = {}
        history[clf]['y_score'] = []
        history[clf]['y_val'] = []
        history[clf]['index_val'] = []
    for fold,(train_index, test_index) in enumerate(tqdm(split_fun)):
        X_train, X_test = train_features[train_index].copy(), train_features[test_index].copy()
        y_train, y_test = y[train_index].copy(), y[test_index].copy()
        if not run_only_cv:
            out = f_outpot[fold]
            get_stg_fun = out['get_stg_fun']
            score_gates = out['score_gates'][0:k]
            topk = out['topk'][0:k]
            col_name = out['col_name'][0:k]
            score_gates = out['score_gates'][0:k]
            history[f"fold_{fold}_score_gates"] = score_gates
            history[f"fold_{fold}_col_name"] = col_name
            history[f"fold_{fold}_fs_time"] = out['time']
        for clf in models:
            start = time.time()
            history[clf][fold] = {}
            estimators = [("Filtering",SelectKBest(get_stg_fun,k=k)), ('clf', models[clf])]
            pipe = Pipeline(estimators)
            pipe.fit(X_train,y_train)
            stop_train = time.time()
            pred = pipe.predict_proba(X_test)
            stop_infer = time.time()
            history[clf][fold] = {}
            try:
                history[clf][fold] = {}
                history[clf][fold]['score'] = get_score(y_test,pred)
                history[clf][fold]['train_time'] = stop_train - start
                history[clf][fold]['infer_time'] =  stop_infer - stop_train
            except Exception as e:
                history[clf][fold]['Exception'] = e
                print(e)
                    
            history[clf]['y_score'].append(pred)
            history[clf]['y_val'].append(y_test)
            
    for clf in models:
        history[clf]['y_score'] = np.concatenate(history[clf]['y_score'])
        history[clf]['y_val'] = np.concatenate(history[clf]['y_val'])
        history[clf]['score']['cv_score'] = get_score(history[clf]['y_val'],history[clf]['y_score'])
        history[clf]['n_splits'] = n_splits
        history[clf]['name_cv'] = name_cv
        history[clf]['k'] = k
    return history

def get_fs_per_fold(df,name,Filtering_Algorithm,train_features,y,train_index,test_index,k=100):
    start = time.time()
    outpot = {}
    X_train, X_test = train_features[train_index].copy(), train_features[test_index].copy()
    y_train, y_test = y[train_index].copy(), y[test_index].copy()
    get_stg_fun = Filtering[Filtering_Algorithm](datasets=name,out_path=f'temp/{Filtering_Algorithm}')
    score_gates = get_stg_fun(X_train,y_train,k=k)
    topk = score_gates.argsort()[::-1][0:k]
    col_name = np.array(df.drop('target',axis=1).columns)[topk]
    score_gates = score_gates[topk] 
    outpot['get_stg_fun'] = get_stg_fun
    outpot['score_gates'] = score_gates
    outpot['topk'] = topk
    outpot['col_name'] = col_name
    outpot['score_gates'] = score_gates
    outpot['time'] = time.time() - start
    return outpot

def run_dataset(datsets_num,Filtering_Algorithm):
    n_jobs = 1
    topk=[1,2,3,4,5,10,15,20,25,30,50,100]
    seedEverything(2022)
    start = time.time()
    ## Init
    os.makedirs(f"temp/{Filtering_Algorithm}/",exist_ok=True)
    df,name = get_specific_df(datsets_num)
    PP = ColumnsPreprocessing(columns=name)
    Filtering_fun = Filtering[Filtering_Algorithm](datasets=name,out_path=f'temp/{Filtering_Algorithm}')
    train_features,y = PP.transform(df)
    skf,n_splits,name_cv = get_cv_split(df)
    run_only_cv = False if n_splits>10 else True
    split_fun = skf.split(train_features, y) if "Folds" in name_cv else skf.split(train_features)
    train_features = train_features.values
    y = y.values
    ## GridSearchCV
    # fun = Filtering[Filtering_Algorithm](datasets=name,out_path=f'temp/{Filtering_Algorithm}')
    seedEverything(2022)
    GridSearchCV_results = run_GridSearchCV(train_features,y,None,name,cv=3)
    ## Filtering_Algorithm
    outpot = None
    if run_only_cv:
        seedEverything(2022)
        outpot = Parallel(n_jobs=int(n_jobs))(delayed(get_fs_per_fold)(df,name,Filtering_Algorithm,train_features,y,train_index,test_index) for fold,(train_index, test_index) in enumerate(tqdm(split_fun))) 
    ## run trian
    seedEverything(2022)
    history = Parallel(n_jobs=int(n_jobs))(delayed(train_kfold)(Filtering_fun,Filtering_Algorithm,k,datsets_num,outpot,PP,**GridSearchCV_results) for k in tqdm(topk))
    clear_output()
    print(time.time()-start)
    with open(f'temp/{Filtering_Algorithm}/{name}_history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=4)

Filtering = {'STG':get_stg_class,
             'new_STG':get_stg_class_new,
            'f_classif':get_SelectFdr_class,
            'mrmr':get_mrmr_class,
            'reliefF':get_reliefF_class,
             'RFE_SVM':get_RFE_SVM_class,
            'FWDT':get_FWDT_class,
            'new_FWDT':get_FWDT_class_new,
            "ensemble":get_ensemble_class,
            "new_ensemble":get_ensemble_class_new}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='pipline')
    parser.add_argument('--filtering',default='f_classif')
    parser.add_argument('--n_job',default=1)
    parser.add_argument('--test',action='store_true')
     
    args = parser.parse_args()
    n_jobs = int(args.n_job)
    if args.test:
        run_dataset(0,'f_classif')
        quit()
    if n_jobs ==1:
         for datsets_num in tqdm(range(63)):
            _=run_dataset(datsets_num,args.filtering)
    else:
        _ = Parallel(n_jobs=int(n_jobs))(delayed(run_dataset)(datsets_num,args.filtering) for datsets_num in tqdm(range(63)))