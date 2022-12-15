import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.utils import shuffle

ttrain_path ='data_train/ag-news/ag-news_TextFooler_bert_x_natural.npy'
cls_model='xgb'
if_natural=False


base_path = ttrain_path.replace(".npy", "").split('/')[0]+'/'+ttrain_path.replace(".npy", "").split('/')[1]
ttrain_file = ttrain_path.replace(".npy", "").split('/')[2]
toutdir =ttrain_path.replace(".npy", "").split('/')[0]
tdataset = ttrain_file.split('_')[0]
tattack=ttrain_file.split('_')[1]
tvictim = ttrain_file.split('_')[2]
tdata = ttrain_file.split('_')[3]
ttypes = ttrain_file.split('_')[4]

x_tpath_n=base_path+'/'+tdataset+"_"+tattack+"_"+tvictim+"_"+"x"+"_"+"natural"+".npy"
x_tpath_s=base_path+'/'+tdataset+"_"+tattack+"_"+tvictim+"_"+"x"+"_"+"sorted"+".npy"
y_tpath=base_path+'/'+tdataset+"_"+tattack+"_"+tvictim+"_"+"y"+".npy"



with open(x_tpath_n,'rb') as f:
    x_ttest_n =np.load(f)
    
with open(x_tpath_s,'rb') as f1:
    x_ttest_s =np.load(f1)
    
with open(y_tpath,'rb') as f2:
    y_ttest =np.load(f2) 
    
lens=len(x_ttest_n)
train_num=lens-1000

if cls_model=='xgb':
    xgb_classifier = xgb.XGBClassifier(
                    max_depth=3,
                    learning_rate=0.34281802,
                    gamma=0.6770816,
                    min_child_weight=2.5520658,
                    max_delta_step=0.71469694,
                    subsample=0.61460966,
                    colsample_bytree=0.73929816,
                    colsample_bylevel=0.87191725,
                    reg_alpha=0.9064181,
                    reg_lambda=0.5686102,
                    n_estimators=29,
                    # silent=0,
                    nthread=4,
                    scale_pos_weight=1.0,
                    base_score=0.5,
                    # missing=None,
    )
    if if_natural:
        xgb_classifier.fit(x_ttest_n[0:train_num], y_ttest[0:train_num])
        predictions=xgb_classifier.predict(x_ttest_n[train_num:])
    else:
        xgb_classifier.fit(x_ttest_s[0:train_num], y_ttest[0:train_num])
        predictions=xgb_classifier.predict(x_ttest_s[train_num:])
else:
    svm_clf = SVC(C=9.0622635,
          kernel='rbf',
          gamma='scale',
          coef0=0.0,
          tol=0.001,
          probability=True,
          max_iter=-1)
    if if_natural:
        svm_clf.fit(x_ttest_n[0:train_num], y_ttest[0:train_num])
        predictions=svm_clf.predict(x_ttest_n[train_num:])
    else:
        svm_clf.fit(x_ttest_s[0:train_num], y_ttest[0:train_num])
        predictions=svm_clf.predict(x_ttest_s[train_num:])



#print detection results
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_ttest[train_num:], predictions, digits=3))
print(confusion_matrix(y_ttest[train_num:], predictions))
tn,fp,fn,tp= confusion_matrix(y_ttest[train_num:], predictions).ravel()
acc= (tn+tp)/(tn+fp+fn+tp)
p = tp/(tp+fp)
r = tp/(tp+fn)

tnr =tn/(tn+fp)
fnr = fn/(fn+tp)

tpr = tp/(tp+fn)

fpr =fp/(tn+fp)
f1_sc = (2*p*r)/(p+r) 
adv_recall = tp/(tp+fn)


print("test_config:",ttrain_path)
print("Accuracy is :",round(acc,4))
print("F1-score is :",round(f1_sc,4))
print("adv recall is:",round(adv_recall,4))
print('*********')
print("TNR is:",round(tnr,4))
print('TPR is:',round(tpr,4))