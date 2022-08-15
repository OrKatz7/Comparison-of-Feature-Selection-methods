import scipy.io
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import glob
import os
import numpy as np
from datetime import datetime
from skfeature.utility.mutual_information import su_calculation
import numpy as np
from math import log
from joblib import Parallel, delayed
import multiprocessing
from skfeature.function.similarity_based import reliefF

def entropy(z):
    cl = np.unique(z)
    hz = 0
    for i in range(len(cl)):
        c = cl[i]
        pz = float(sum(z == c)) / len(z)
        hz = hz + pz * log(pz, 2)
    hz = -hz
    return hz


def infogain(x, y):
    '''
        x: features (data)
        y: output (classes)
    '''
    info_gains_par = np.ones(x.shape[1]) # features of x
    info_gains = np.ones(x.shape[1]) # features of x
    nrows = x.shape[0]
    # calculate entropy of the data *hy* with regards to class y

    hy = entropy(y)
    info_gains *= hy

    # ====================== Non-Parallel ================================

#     for i in tqdm(range(len(info_gains))):
#         xi_unique = np.unique(x[:, i])
#         #xi_unique = unique(array(x[:,i])[:,0])
#         for j in range(len(xi_unique)):
#             indexi = (xi_unique[j] == x[:, i])
#             #indexi = (xi_unique[j] == array(x[:,i])[:,0])
#             info_gains[i] -= entropy(y[indexi])*(sum(indexi)/nrows)

    # ======================   Parallel   ================================
    def calc_entropy_for_col(i):
        entropy_sum = 0
        xi_unique = np.unique(x[:, i])
        for j in range(len(xi_unique)):
            indexi = (xi_unique[j] == x[:, i])
            entropy_sum += entropy(y[indexi])*(sum(indexi)/nrows)
        return entropy_sum
    entropy_sum = Parallel(n_jobs=6)(delayed(calc_entropy_for_col)(i) for i in range(len(info_gains)))
    
    info_gains = info_gains - np.array(entropy_sum)
    
    # ======================         ================================
    
    return info_gains


def interact(x, y):
    features = x
    classes = y
    scores = [su_calculation(features[:, feature_i], classes) for feature_i in range(features.shape[1])]
    scores = np.array(scores)
    return scores


def relieff(x, y):
    reliefF_ranking = reliefF.reliefF(x, y)
    reliefF_scores = reliefF_ranking / len(reliefF_ranking)
    return reliefF_scores


def ensemble_of_filters(x, y):
    infogain_scores = infogain(x, y)
    interact_scores = interact(x, y)
    reliefF_scores = relieff(x, y)
    
    ensemble = (infogain_scores + interact_scores + reliefF_scores) / 3
    
    return ensemble


def ensemble_of_filters_new(x, y):
    # ### this didnt improve the results ###
    # w = [0.2, 0.2, 0.6]
    # infogain_scores = infogain(x, y)
    # interact_scores = interact(x, y)
    # reliefF_scores = relieff(x, y)
    # ensemble = (infogain_scores*w[0] + interact_scores*w[1] + reliefF_scores*w[2]) / 3
    
    infogain_scores = np.array(infogain(x, y))
    infogain_order = infogain_scores.argsort()
    infogain_ranks = infogain_order.argsort()
    
    interact_scores = np.array(interact(x, y))
    interact_order = interact_scores.argsort()
    interact_ranks = interact_order.argsort()
    
    reliefF_scores = np.array(relieff(x, y) )
    reliefF_order = reliefF_scores.argsort()
    reliefF_ranks = reliefF_order.argsort()
    
    ensemble = infogain_ranks + interact_ranks + reliefF_ranks
    
    return ensemble