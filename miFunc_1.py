
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch import int64
import torch.nn as torch
import os
from copy import copy
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import re

def mask_string(orig_str,tokenizer):
    pattern = re.compile(r'\s')
    result = pattern.findall(orig_str)
    r_idx=0
    r_lo=[0]*(len(result)+2)
    r_lo[-1]=-1
    for s_idx in range(len(orig_str)):
        if orig_str[s_idx]==result[r_idx]:
            r_lo[r_idx+1]=s_idx
            r_idx+=1
            if r_idx>=len(result):
                break
    mask_str=[]
    for idx in range(len(r_lo)-1):
        if r_lo[idx]==0:
            # os_='[MASK]'+orig_str[r_lo[idx+1]:]
            os_='<mask>'+orig_str[r_lo[idx+1]:]
            mask=orig_str[r_lo[idx]:r_lo[idx+1]]
        elif r_lo[idx+1]==-1:
            # os_=orig_str[0:r_lo[idx]]+' [MASK]'
            os_=orig_str[0:r_lo[idx]]+' <mask>'
            mask=orig_str[r_lo[idx]+1:]
        else:
            # os_=orig_str[0:r_lo[idx]]+' [MASK]'+orig_str[r_lo[idx+1]:]
            os_=orig_str[0:r_lo[idx]]+' <mask>'+orig_str[r_lo[idx+1]:]
            mask=orig_str[r_lo[idx]+1:r_lo[idx+1]]
        if len(mask)>0:
            mask_str.append({})
            mask_str[-1]['seq']=os_
            mask_str[-1]['mask']=mask
    if len(mask_str)>512:
        mask_str_t=mask_str[:512]
    else:
        mask_str_t= mask_str
    mask_str_fi=[]
    for idx in range(len(mask_str_t)):
        mask_str_tokenize = tokenizer(mask_str_t[idx]['seq'],return_tensors="pt",padding=True, truncation=True).to(device)
        mask_token_index = (mask_str_tokenize.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        if len(mask_token_index)!=0: 
            mask_str_t[idx]['mask_token_index'] =mask_token_index         
            mask_str_fi.append(mask_str_t[idx])
    return mask_str_fi

def rand_mask_string(raw_str,tokenizer,ma_fre):
    mask_str=[]
    raw_str_list = raw_str.split(' ')
    ran_index =np.random.choice(len(raw_str_list),int(len(raw_str_list)*ma_fre))
    for idx in range(len(ran_index)):
        new_str_list=copy(raw_str_list)
        new_str_list[ran_index[idx]]='<mask>'
        # new_str_list[ran_index[idx]]='[MASK]'
        new_str=' '.join(new_str_list)
        mask=raw_str_list[ran_index[idx]]
        mask_str.append({})
        mask_str[-1]['seq']=new_str
        mask_str[-1]['mask']=mask
    if len(mask_str)>512:
        mask_str_t=mask_str[:512]
    else:
        mask_str_t= mask_str
    mask_str_fi=[]
    for idx in range(len(mask_str_t)):
        mask_str_tokenize = tokenizer(mask_str_t[idx]['seq'],return_tensors="pt",padding=True, truncation=True).to(device)
        mask_token_index = (mask_str_tokenize.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        if len(mask_token_index)!=0: 
            mask_str_t[idx]['mask_token_index'] =mask_token_index         
            mask_str_fi.append(mask_str_t[idx])
    return mask_str_fi

def mlm_ummask(model,tokenizer,masked_str_idx,k=3):
    pattern_error=re.compile(r'\[[a-zA-Z]+\]')
    token=[]
    inputs = tokenizer(masked_str_idx['seq'], return_tensors="pt",padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        values,indices= torch.topk(logits[0, mask_token_index],k=k,dim=-1)
        for i in range(k):
            token_tmp=tokenizer.decode(torch.flatten(indices)[i])
            token_tmp=token_tmp.replace(' ','')
            ret_error = pattern_error.search(token_tmp)
            if token_tmp.find('##') ==0 or ret_error is not None:
                token_tmp =' '
            token.append(token_tmp)
    return token

def masked_str_token(masked_str,model,tokenizer):
    masked_str_tokens=[]
    for idx in range(len(masked_str)):
        masked_str_tokens.append(mlm_ummask(model,tokenizer,masked_str[idx]))
    return masked_str_tokens


def get_mlm_um_sen(masked_str,masked_str_tokens,tn):
    mlm_um_sen1=[]
    for idx in range(len(masked_str)):
        # fill_sen = masked_str[idx]['seq']
        fill_tokens = masked_str_tokens[idx]
        # um_sen =[]
        for idy in range(tn):
            # um_str1=masked_str[idx]['seq'].replace('[MASK]',fill_tokens[idy])
            um_str1=masked_str[idx]['seq'].replace('<mask>',fill_tokens[idy])
            mlm_um_sen1.append(um_str1)
    return mlm_um_sen1



#validation
def gap_infer_acc_f1 (train_metric, test_metric):
    n = len(train_metric)
    acc=0
    gap_a=0
    gap_f=0
    f1=0
    y_true = np.append(np.zeros(n,np.int64),np.ones(n,np.int64))
    train_test_meritc = np.append(train_metric,test_metric)
    train_test_meritc_sort = np.sort(train_test_meritc)
    for trs in train_test_meritc_sort:
        y_pred_t = np.where(train_test_meritc>=trs,0,1)
        gap_t = trs
        tn,fp,fn,tp= confusion_matrix(y_true,y_pred_t).ravel()
        acc_t = (tn+tp)/(tn+fp+fn+tp)
        if acc_t >=acc:
            acc=acc_t
            gap_a = gap_t
        else:
            acc=acc
            gap_a = gap_a
            
        if ((tp+fp)==0) or ((tp+fn)==0):
            continue
        else:
            p = tp/(tp+fp)
            r =tp/(tp+fn)
        f1_t = (2*p*r)/(p+r)
        if f1_t>=f1:
            f1=f1_t
            gap_f=gap_t
        else:
            f1=f1
            gap_f=gap_f
    
    #\tau        
    y_pred_a = np.where(train_test_meritc>=gap_a,0,1)
    print(confusion_matrix(y_true,y_pred_a))
    
    y_pred_f = np.where(train_test_meritc>=gap_f,0,1)
    print(confusion_matrix(y_true,y_pred_f))
    return gap_a,round(acc,4),gap_f,round(f1,4)

  

if __name__=='__main__':
    print('fighting')