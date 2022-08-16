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
