print("Start")
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
import os
from tqdm.auto import tqdm
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
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
from tqdm import tqdm

class runner:
    def __init__(self,
                 datsets_num=0,
                 fs_calss = get_stg_class,
                 Filtering_Algorithm = 'stg',
                 fs_args = {'out_path':'output/stg',},
                 knn_args = {"n_neighbors": 10},
                 rf_args = {"n_estimators":100},
                 lr_args = {"C":1e5},
                 SVC_args = {"C":2,"probability":True},
                 NB_args = {"alpha":1},
                 run_first_time  = True,
                 run_grid = True
                ):
        self.datsets_num = datsets_num
        self.fs_args = fs_args
        self.knn_args = knn_args
        self.rf_args = rf_args
        self.lr_args = lr_args
        self.SVC_args = SVC_args
        self.NB_args = NB_args
        self.fs_calss = fs_calss
        self.Filtering_Algorithm = Filtering_Algorithm
        df,name = get_specific_df(self.datsets_num)
        self.name = name
        self.df = df
        if run_first_time:
            self.run_first_time()
        if run_grid:
            self.results = self.run_GridSearchCV()
            self.CTKF = ColumnsTrainKfold(**self.results)
            self.train_one_fold = TrainOneFold(**self.results)
            
    def run_GridSearchCV(self):
        if os.path.exists(f'output/Preprocessing/{self.name}.pickle'):
            with open(f'output/Preprocessing/{self.name}.pickle', 'rb') as handle:
                results = pickle.load(handle)
            return results
        results = {}
        for n,params in tqdm(zip(grid_parameters_name,grid_parameters)):
            clf = params['clf'][0]
            params2 = params.copy()
            params2.pop('clf')
            estimators = [('ColumnsPreprocessing', self.CPP), ("SelectKBest",SelectKBest(self.get_stg_fun,k=100)), ('clf', clf)]
            pipe = Pipeline(estimators)
            grid = GridSearchCV(pipe, param_grid=params2, cv=3)
            _=grid.fit(self.df,self.y)
            results[n] = {}
            for row in grid.best_params_:
                results[n][row.split("__")[-1]] = grid.best_params_[row]
        with open(f'output/Preprocessing/{self.name}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=4)
        return results
            
    def run_first_time(self):
        df,name = get_specific_df(self.datsets_num)
        self.CPP = ColumnsPreprocessing(columns=name)
        self.train_features,self.y = self.CPP.transform(df)
        self.get_stg_fun = self.fs_calss(datasets=name,**self.fs_args)
        fs_time = time.time()
        new_train_features = SelectKBest(self.get_stg_fun,k=100).fit_transform(self.train_features.values,self.y)
        fs_time = time.time()-fs_time
        self.fs_time = fs_time
        self.CTKF = ColumnsTrainKfold(
                      knn_args = self.knn_args,
                      rf_args = self.rf_args,
                      lr_args = self.lr_args,
                      SVC_args = self.SVC_args,
                      NB_args = self.NB_args)
        
        self.train_one_fold = TrainOneFold(
                      knn_args = self.knn_args,
                      rf_args = self.rf_args,
                      lr_args = self.lr_args,
                      SVC_args = self.SVC_args,
                      NB_args = self.NB_args)
        
        estimators = [('ColumnsPreprocessing', self.CPP), ("SelectKBest",SelectKBest(self.get_stg_fun,k=100)), ('train', self.CTKF)]
        pipe = Pipeline(estimators)
        _ = pipe.fit_transform(self.df,self.y)
        
    def run_loop(self,topk=[1,2,3,4,5,10,15,20,25,30,50,100]):
        self.history_df = []
        for k in topk:
            estimators = [('ColumnsPreprocessing', self.CPP), ("SelectKBest",SelectKBest(self.get_stg_fun,k=k)), ('train', self.CTKF)]
            train_features,y = self.CPP.transform(self.df)
            score_gates = self.get_stg_fun(train_features,y)
            topk = score_gates.argsort()[::-1][0:k]
            col_name = np.array(self.df.drop('target',axis=1).columns)[topk]
            score_gates = score_gates[topk]
            pipe = Pipeline(estimators)
            history = pipe.fit_transform(self.df,self.y)
            for m in history.keys():
                for metric in ['ACC','MCC','AUC','PR-AUC']:
                    self.history_df.append({
                                       "Dataset Name": self.name,
                                        "Number of samples":len(self.df),
                                        "Original Number of features":self.df.shape[1],
                                        "Filtering Algorithm":self.Filtering_Algorithm,
                                        "Learning algorithm":m,
                                        "Number of features selected (K)":k,
                                        "CV Method": history[m]['name_cv'],
                                        "Folds":history[m]['n_splits'],
                                        "List of Selected Features Names (Long STRING)":col_name,
                                        "Selected Features scores":score_gates,
                                        "Measure Type":metric,
                                        "Measure Value":history[m]['score'][metric],
                                        "Learning algorithm train time":history[m]['train_time'],
                                        "Learning algorithm predict proba time":history[m]['infer_time'],
                                        "Filtering algorithm time":self.fs_time,
                                      })
        history_summary= pd.DataFrame(self.history_df)
        self.history_summary = history_summary
        return history_summary
    
    def run_loop_by_fold(self,topk=[1,2,3,4,5,10,15,20,25,30,50,100]):
        self.history_df = []
        train_features,y = self.CPP.transform(self.df)
        train_features2 = train_features.values
        skf,n_splits,name_cv = get_cv_split(self.df)
        split_fun = skf.split(train_features, y) if "Folds" in name_cv else skf.split(train_features)
        t = None
        t_p = None
        for fold,(train_index, test_index) in enumerate(tqdm(split_fun)):
            self.get_stg_fun = self.fs_calss(datasets=f"fold_{fold}_{self.name}",**self.fs_args)
            X_train, X_test = train_features2[train_index].copy(), train_features2[test_index].copy()
            y_train, y_test = y[train_index].copy(), y[test_index].copy()
            for k in topk:
                start = time.time()
                history = {}
                estimators = [("SelectKBest",SelectKBest(self.get_stg_fun,k=k)), ('train', self.train_one_fold)]
                pipe = Pipeline(estimators)
                _ = pipe.fit(X_train,y_train)
                
                score_gates = self.get_stg_fun(X_train,y_train)
                topk = score_gates.argsort()[::-1][0:k]
                col_name = np.array(self.df.drop('target',axis=1).columns)[topk]
                score_gates = score_gates[topk]
                
                if t is None:
                    t = time.time() - start
                start = time.time()
                y_score = pipe.predict_proba(X_test)
                if t_p is None:
                    t_p = time.time() - start
                y_pred = pipe.predict(X_test)
                for k in y_score:
                    history[k] = {}
                    history[k]['score'] = get_score(y_test,y_score[k])
                for m in history.keys():
                    for metric in ['ACC','MCC','AUC','PR-AUC']:
                        self.history_df.append({
                                           "Dataset Name": self.name,
                                            "Number of samples":len(self.df),
                                            "Original Number of features":self.df.shape[1],
                                            "Filtering Algorithm":self.Filtering_Algorithm,
                                            "Learning algorithm":m,
                                            "Number of features selected (K)":k,
                                            "CV Method": name_cv,
                                            "fold":fold,
                                            "List of Selected Features Names (Long STRING)":col_name,
                                            "Selected Features scores":score_gates,
                                            "Measure Type":metric,
                                            "Measure Value":history[m]['score'][metric],
                                            "Learning algorithm train time":t,
                                            "Learning algorithm predict proba time":t_p,
                                            "Filtering algorithm time":self.fs_time,
                                          })
        history_summary= pd.DataFrame(self.history_df)
        self.history_summary = history_summary
        return history_summary
    
    def run_best_conf(self,topk=[20]):
        self.results = self.run_GridSearchCV()
        self.CTKF = ColumnsTrainKfold(                   
                                    use_smote = True,
                                    use_pca = True,
                                    **self.results)
        self.history_df = []
        for k in topk:
            estimators = [('ColumnsPreprocessing', self.CPP), ("SelectKBest",SelectKBest(self.get_stg_fun,k=k)), ('train', self.CTKF)]
            train_features,y = self.CPP.transform(self.df)
            score_gates = self.get_stg_fun(train_features,y)
            topk = score_gates.argsort()[::-1][0:k]
            col_name = np.array(self.df.drop('target',axis=1).columns)[topk]
            score_gates = score_gates[topk]
            pipe = Pipeline(estimators)
            history = pipe.fit_transform(self.df,self.y)
            for m in history.keys():
                for metric in ['ACC','MCC','AUC','PR-AUC']:
                    self.history_df.append({
                                       "Dataset Name": self.name,
                                        "Number of samples":len(self.df),
                                        "Original Number of features":self.df.shape[1],
                                        "Filtering Algorithm":self.Filtering_Algorithm,
                                        "Learning algorithm":m,
                                        "Number of features selected (K)":k,
                                        "CV Method": history[m]['name_cv'],
                                        "Folds":history[m]['n_splits'],
                                        "List of Selected Features Names (Long STRING)":col_name,
                                        "Selected Features scores":score_gates,
                                        "Measure Type":metric,
                                        "Measure Value":history[m]['score'][metric],
                                        "Learning algorithm train time":history[m]['train_time'],
                                        "Learning algorithm predict proba time":history[m]['infer_time'],
                                        "Filtering algorithm time":self.fs_time,
                                      })
        history_summary= pd.DataFrame(self.history_df)
        self.history_summary = history_summary
        return history_summary
    
    def plot_metric(self,metric):
        plt.figure(figsize=[20,5])
        for i,mini_df in self.history_summary.groupby("Learning algorithm"):
            mini_df = mini_df.sort_values("Number of features selected (K)")
            mini_df = mini_df.set_index("Number of features selected (K)")
            mini_df = mini_df[mini_df['Measure Type']==metric]
            plt.plot(mini_df[['Measure Value']],label=i,marker='o')
        plt.legend()
        plt.ylabel(f'{metric}', fontsize=20)
        plt.xlabel(r'Number of features selected (K)', fontsize=20)
        plt.show()
        
    def save_results(self,path2save):
        self.history_summary.to_csv(f"{path2save}/{self.name}",index=False)
        
    def load_results(self,path2load):
        self.history_summary = pd.read_csv(f"{path2load}/{self.name}")
        
        
def Parallel_run(datsets_num,get_stg_class,Filtering_Algorithm):
    try:
        os.makedirs(f'output/{Filtering_Algorithm}',exist_ok = True)
        seedEverything(2022)
        run = runner(                 
                datsets_num=datsets_num,
                fs_calss = get_stg_class,
                Filtering_Algorithm = Filtering_Algorithm,
                fs_args = {'out_path':f'output/{Filtering_Algorithm}',},
                knn_args = {"n_neighbors": 10},
                rf_args = {"n_estimators":100},
                lr_args = {"C":1e5},
                SVC_args = {"C":2,"probability":True},
                NB_args = {"alpha":1},
                run_first_time = True,
                run_grid = True)
        history_summary = run.run_loop(topk=[1,2,3,4,5,10,15,20,25,30,50,100])
        run.save_results(f'output/{Filtering_Algorithm}')
        return True
    except Exception as e:
        print(e)
        return False
    
def Parallel_run_best(datsets_num):
    # try:
        seedEverything(2022)
        load_run = runner(                 
                datsets_num=datsets_num,
                run_first_time = False,
                run_grid = False)
        history_summary = []
        for d in Filtering:
            load_run.load_results(f'output/{d}')
            history_summary.append(load_run.history_summary) 
        history_summary = pd.concat(history_summary)
        metric = 'AUC'
        history_summary = history_summary[history_summary['Measure Type']==metric]
        best = history_summary[history_summary['Measure Value'] == history_summary['Measure Value'].max()]
        Filtering_Algorithm = list(best["Filtering Algorithm"])[0]
        get_stg_class = Filtering[Filtering_Algorithm]
        k = list(best['Number of features selected (K)'])[0]
        print(list(best['Measure Value'])[0])
        os.makedirs(f'output/aug_best',exist_ok = True)
        run = runner(                 
                datsets_num=datsets_num,
                fs_calss = get_stg_class,
                Filtering_Algorithm = Filtering_Algorithm,
                fs_args = {'out_path':'output/aug_best',},
                knn_args = {"n_neighbors": 10},
                rf_args = {"n_estimators":100},
                lr_args = {"C":1e5},
                SVC_args = {"C":2,"probability":True},
                NB_args = {"alpha":1},
                run_first_time = True,
                run_grid = True)
        history_summary = run.run_best_conf(topk=[k])
        history_summary = history_summary[history_summary['Measure Type']==metric]
        best = history_summary[history_summary['Measure Value'] == history_summary['Measure Value'].max()]
        print(list(best['Measure Value'])[0])
        run.save_results(f'output/aug_best')
        return True
    # except Exception as e:
    #     print(e)
    #     return False

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
    parser.add_argument('--filtering',default='STG')
    parser.add_argument('--n_job',default=6)
    parser.add_argument('--backend',default='loky')
    parser.add_argument('--run_only_best',action='store_true')
     
    args = parser.parse_args()
    if not args.run_only_best:
        Filtering_Algorithm = args.filtering
        fun = Filtering[Filtering_Algorithm]
        log_error = Parallel(n_jobs=int(args.n_job),backend=args.backend,verbose=0)(delayed(Parallel_run)(datsets_num,fun,Filtering_Algorithm) for datsets_num in tqdm(range(63)))
    else:
        log_error = Parallel(n_jobs=int(args.n_job),backend=args.backend,verbose=0)(delayed(Parallel_run_best)(datsets_num) for datsets_num in tqdm(range(63)))

