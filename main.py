#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import itertools
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report , confusion_matrix

import warnings
warnings.filterwarnings('ignore')

def read_data():
    df=pd.read_csv('framingham.csv')
    return df

def data_preprocessing(df):
    df.drop('education',axis=1,inplace=True)
    df.dropna(axis=0,inplace=True)
    
    #train_test_split
    dataX=df.iloc[:,:-1]
    datay=df.iloc[:,-1]
    X_train, X_test, y_train, y_test =train_test_split(dataX,datay,test_size=0.3,random_state=0)

    ##Over sampling minority 
    sm = SMOTE(random_state=12, sampling_strategy = 'minority')
    x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    X_train=pd.DataFrame(data=x_train_res)
    y_train=pd.DataFrame(data= y_train_res)

    return (X_train, X_test, y_train, y_test,dataX,datay)

def logistic_regression(X_train, y_train):

    lr=LogisticRegression().fit(X_train, y_train)
    return lr
def predict_probability(model,X_test,THRESHOLD):
    y_pred=np.where(model.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)
    return y_pred

def cv_validation(fitted_model,datasetX,datasety,n_folds=4,THRESHOLD=0.5,over='Dataset'):  
    # prepare cross validation
    kfold = KFold(n_folds, True, 1)
    # enumerate splits
    scores=[]
    X_train=datasetX
    y_train=datasety
    for train, test in kfold.split(X_train,y_train):
        #print('train: %s, test: %s' % (X_train.iloc[train,:].shape, X_train.iloc[test,:].shape))
        X_train_sub,X_test_sub,y_train_sub,y_test_sub=X_train.iloc[train,:], X_train.iloc[test,:],y_train.iloc[train],y_train.iloc[test]
        #print(X_train_sub.shape,X_test_sub.shape,y_train_sub.shape,y_test_sub.shape)
        fitted_model.fit(X_train_sub,y_train_sub)
        #if fitted_model== 'lr':
        y_pred=np.where(fitted_model.predict_proba(X_test_sub)[:,1] > THRESHOLD, 1, 0)
        #else :
        #y_pred=fitted_model.predict(X_test_sub)
        scores.append(metrics.accuracy_score(y_test_sub,y_pred))
        #print(metrics.accuracy_score(y_test_sub,y_pred))
    print( 'CV over {}'.format(over) , np.mean(scores))
    return np.mean(scores)

def print_Confusion_Matrix_classification_report(lr,y_pred,y_test):
    print('coefficients corresponding to features')
    print(pd.DataFrame(np.array(lr.coef_),columns=X_train.columns))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred,labels=[0, 1]))
    tn, fp, fn, tp=confusion_matrix(y_test,y_pred,labels=[0, 1]).ravel()
    print('True negatives',tn)
    print('False postives',fp)
    print('False negatives',fn)
    print('True positives',tp)

    
def row_probability(clf,attrs,THRESHOLD=0.5):
    r_pred=np.where(clf.predict_proba(attrs)[:,1] > THRESHOLD, 1, 0)
    #r_pred = clf.predict_proba(attrs)
    return r_pred
    

# In[ ]:





# In[ ]:




