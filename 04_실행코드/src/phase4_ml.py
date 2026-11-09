"""Small declared search space, grouped evaluation, and empirical score policies."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss, confusion_matrix)
from .config import FEATURES

PROCESS = [c for c in FEATURES if c.startswith(('force_','torque_','tension_',
           'roll_speed_','motor_power_','gap_','reduction_'))]
CONFIGS = [
    dict(id='rf51',family='RandomForest',features=FEATURES,params=dict(n_estimators=140,max_depth=18,min_samples_leaf=2,class_weight='balanced_subsample')),
    dict(id='rf36',family='RandomForest',features=PROCESS,params=dict(n_estimators=180,max_depth=24,min_samples_leaf=2,class_weight=None)),
    dict(id='et51',family='ExtraTrees',features=FEATURES,params=dict(n_estimators=140,max_depth=24,min_samples_leaf=2,class_weight='balanced',max_features='sqrt')),
    dict(id='et36',family='ExtraTrees',features=PROCESS,params=dict(n_estimators=180,max_depth=None,min_samples_leaf=4,class_weight=None,max_features=.7)),
    dict(id='hgb51',family='HistGradientBoosting',features=FEATURES,params=dict(max_iter=120,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,class_weight='balanced')),
    dict(id='hgb36',family='HistGradientBoosting',features=PROCESS,params=dict(max_iter=160,max_leaf_nodes=31,learning_rate=.05,l2_regularization=5.,class_weight=None)),
    dict(id='lr51',family='LogisticRegression',features=FEATURES,params=dict(C=.1,class_weight='balanced',max_iter=1500)),
    dict(id='lr36',family='LogisticRegression',features=PROCESS,params=dict(C=1.,class_weight=None,max_iter=1500)),
]


def make_pipeline(config):
    family, params = config['family'], dict(config['params'])
    if family in ('RandomForest','ExtraTrees'):
        estimator = (RandomForestClassifier if family=='RandomForest' else ExtraTreesClassifier)(
            **params, random_state=42,n_jobs=4)
    elif family == 'HistGradientBoosting':
        # Disable automatic row-based early stopping across task boundaries.
        estimator = HistGradientBoostingClassifier(**params,random_state=42,early_stopping=False)
    else:
        estimator = LogisticRegression(**params,random_state=42)
    steps = [('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if family == 'LogisticRegression':
        steps.append(('scaler',StandardScaler()))
    return Pipeline(steps+[('estimator',estimator)])


def score_metrics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),normal=int(tn+fp),anomalies=int(fn+tp),
        precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
        roc_auc=float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
        average_precision=float(average_precision_score(y,scores)) if (y==1).any() else None,
        false_alarms=int(fp),missed_anomalies=int(fn),
        false_alarms_per_1000_normal=float(fp/max(1,tn+fp)*1000),
        misses_per_1000_anomalies=float(fn/max(1,fn+tp)*1000),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        brier=float(brier_score_loss(y,scores)),log_loss=float(log_loss(y,scores,labels=[0,1])))


def recall_threshold(y, scores, target=.70):
    """Largest score cutoff meeting pooled recall, hence minimal pooled FP.

    Uses only supplied validation labels. Ties at the cutoff stay positive.
    """
    y, scores = np.asarray(y),np.asarray(scores,dtype=float)
    if not 0 < target <= 1 or not len(scores) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('유효한 점수와 목표 Recall(0 초과 1 이하)이 필요합니다.')
    positive = np.sort(scores[y==1])
    if not len(positive):
        raise ValueError('임계값 선택에 이상 라벨이 필요합니다.')
    return float(positive[len(positive)-int(np.ceil(target*len(positive)))])


def threshold_curve(y,scores):
    """All unique cutoffs plus 0 and 1, O(n log n), including confusion counts."""
    import pandas as pd
    y,scores=np.asarray(y),np.asarray(scores)
    order=np.argsort(scores); sorted_scores=scores[order]; sorted_y=y[order]
    thresholds=np.unique(np.r_[0.,scores,1.])
    positions=np.searchsorted(sorted_scores,thresholds,side='left')
    cpos=np.r_[0,np.cumsum(sorted_y==1)]; cneg=np.r_[0,np.cumsum(sorted_y==0)]
    fn,tn=cpos[positions],cneg[positions]; tp,fp=cpos[-1]-fn,cneg[-1]-tn
    precision=tp/np.maximum(tp+fp,1);recall=tp/max(1,cpos[-1])
    return pd.DataFrame(dict(threshold=thresholds,precision=precision,recall=recall,
        f1=2*tp/np.maximum(2*tp+fp+fn,1),false_alarms_per_1000_normal=fp/max(1,cneg[-1])*1000,
        misses_per_1000_anomalies=fn/max(1,cpos[-1])*1000,tn=tn,fp=fp,fn=fn,tp=tp))
