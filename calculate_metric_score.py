from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_curve, auc, average_precision_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

def get_score(y_true, y_prob):
    '''
    metric = {'ACC', 'MCC', 'AUC', 'PR-AUC'}
    ''';
    scores = {}
    
    if len(y_prob.shape) == 2:
        y_pred = np.argmax(y_prob, 1)
    if len(y_prob.shape) == 1:
        y_pred = np.argmax(y_prob, 0)
    # print('y_prob', y_prob)
    # print('y_pred', y_pred)
    
    if len(y_true.shape) == 0:
        n_classes = 1
    else:
        n_classes = max(y_true)+1#len(set(y_true))
        
    is_multiclass = True if n_classes > 2 else False
    
    if len(y_true.shape) == 0 and len(y_pred.shape) == 0:
        scores['MCC'] = matthews_corrcoef([y_true], [y_pred])  
        scores['ACC'] = accuracy_score([y_true], [y_pred])
        scores['AUC'] = 1.0 if y_true == y_pred else 0.0
        scores['PR-AUC'] = 1.0 if y_true == y_pred else 0.0
        return scores

    scores['MCC'] = matthews_corrcoef(y_true, y_pred)  
    scores['ACC'] = accuracy_score(y_true, y_pred)
                
    y_true_cat = label_binarize(y_true, classes=list(range(n_classes)))
    if is_multiclass:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_cat[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_cat.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        scores['AUC'] =  roc_auc["micro"]
    else:
        fpr, tpr,  _= roc_curve(y_true, y_prob[:,1])
        scores['AUC'] = auc(fpr, tpr)
            
    y_true_cat = label_binarize(y_true, classes=list(range(n_classes)))
    if is_multiclass:
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_cat[:, i], y_prob[:, i])
            average_precision[i] = average_precision_score(y_true_cat[:, i], y_prob[:, i])

        precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_cat.ravel(), y_prob.ravel())
        average_precision["micro"] = average_precision_score(y_true_cat, y_prob, average="micro")
        scores['PR-AUC'] = average_precision["micro"]
    else:
        scores['PR-AUC'] = average_precision_score(y_true, y_prob[:,1],  average='micro')
        
    return scores



# def get_score(y_true, y_prob):
#     '''
#     metric = {'ACC', 'MCC', 'AUC', 'PR-AUC'}
#     ''';
#     scores = {}
    
    
#     y_pred = np.argmax(y_prob, 1)
#     n_classes = len(set(y_true))
#     is_multiclass = True if n_classes > 2 else False
    
#     scores['MCC'] = matthews_corrcoef(y_true, y_pred)    
#     scores['ACC'] = accuracy_score(y_true, y_pred)
    
#     y_true_cat = label_binarize(y_true, classes=list(range(n_classes)))
#     if is_multiclass:
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         for i in range(n_classes):
#             fpr[i], tpr[i], _ = roc_curve(y_true_cat[:, i], y_prob[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])

#         # Compute micro-average ROC curve and ROC area
#         fpr["micro"], tpr["micro"], _ = roc_curve(y_true_cat.ravel(), y_prob.ravel())
#         roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#         scores['AUC'] =  roc_auc["micro"]
#     else:
#         fpr, tpr,  _= roc_curve(y_true, y_prob[:,1])
#         scores['AUC'] = auc(fpr, tpr)
            
#     y_true_cat = label_binarize(y_true, classes=list(range(n_classes)))
#     if is_multiclass:
#         precision = dict()
#         recall = dict()
#         average_precision = dict()
#         for i in range(n_classes):
#             precision[i], recall[i], _ = precision_recall_curve(y_true_cat[:, i], y_prob[:, i])
#             average_precision[i] = average_precision_score(y_true_cat[:, i], y_prob[:, i])

#         precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_cat.ravel(), y_prob.ravel())
#         average_precision["micro"] = average_precision_score(y_true_cat, y_prob, average="micro")
#         scores['PR-AUC'] = average_precision["micro"]
#     else:
#         scores['PR-AUC'] = average_precision_score(y_true, y_prob[:,1],  average='micro')
        
#     return scores

# if __name__ == "__main__":
#     import pandas as pd
#     from sklearn import preprocessing

#     csv = 'data/ARFF/Breast.csv'
#     # csv = 'data/ARFF/Leukemia_3c.csv'

#     df = pd.read_csv(csv)

#     le = preprocessing.LabelEncoder()
#     le.fit(list(df['target']))
#     df['target'] = le.transform(df['target'])     

#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import accuracy_score
#     from tqdm import tqdm

#     _train_df, _test_df = train_test_split(df, test_size=0.2, stratify=df['target'])

#     _y_train = _train_df.pop('target')
#     _y_train = _y_train.to_numpy()
#     _X_train = _train_df.to_numpy()

#     _y_test = _test_df.pop('target')
#     _y_test = _y_test.to_numpy()
#     _X_test = _test_df.to_numpy()

#     _clf = RandomForestClassifier(random_state=42)
#     _clf.fit(_X_train, _y_train)
#     _y_pred = _clf.predict(_X_test)
#     _y_prob = _clf.predict_proba(_X_test)

#     _acc = accuracy_score(_y_test, _y_pred)
#     print(_acc)
    
#     print(get_score(_y_test, _y_prob, mat))