from victim import predict_class
import numpy as np
import torch
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def euq_label(orig_str,um_sen_label,model_cls,tokenizer_cls,tn,huggingface_model):
    orig_str2list=[]
    orig_str2list.append(orig_str)
    _,_,orig_str_pre_label = predict_class(orig_str2list,model_cls,tokenizer_cls,huggingface_model)
    euq_label_t=[]
    for idx in range(len(um_sen_label)):
        if  int(orig_str_pre_label[0])!=int(um_sen_label[idx]):
            euq_label_t.append(0)
        else:
            euq_label_t.append(1)
    euq_label=[]
    for idy in range(0,len(euq_label_t),3):
        euq_label.append([euq_label_t[idy],euq_label_t[idy+1],euq_label_t[idy+2]])  
    return euq_label    

def mean_mean_label(all_label):
    sen_n=len(all_label)
    mean_label=np.zeros(sen_n)
    for idx in range(sen_n):
        mean_label[idx]=np.mean(all_label[idx])
    return mean_label


def compute_score_difference(orig_str_sorce,um_sentence_socres):
    n_classes = len(orig_str_sorce[0])
    orig_str_sorce=orig_str_sorce.cpu().numpy()[0]
    predicted_class = np.argmax(orig_str_sorce) 
    class_confi_sorce =orig_str_sorce[predicted_class]
    order = [predicted_class] + [i for i in range(n_classes) if i!=predicted_class]
    data_fin = torch.index_select(um_sentence_socres, 1, torch.LongTensor(order))

    data_socre_difference =data_fin[:, :1].flatten() - torch.max(data_fin[:, 1:], dim=1).values.flatten()
    return data_socre_difference.reshape(-1,1)

def compute_score_difference_padding(orig_str_sorce,um_sentence_socres,target_size=1536):
    
    data_socre_difference= compute_score_difference(orig_str_sorce,um_sentence_socres)
    data_size = min(1536, data_socre_difference.shape[0])

    target = torch.zeros(target_size, 1).to(device)
    target[:data_size, :] = data_socre_difference
    #tensor
    return target
