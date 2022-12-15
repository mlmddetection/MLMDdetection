
from lib2to3.pgen2 import token

# from shutil import register_unpack_format
# from statistics import mode
import numpy as np
import pandas as pd
import textattack
import torch
import re
import os
import torch.nn as nn
from sklearn.svm import SVC
from textattack.models.helpers import (GloveEmbeddingLayer,
                                       LSTMForClassification,
                                       WordCNNForClassification)
from textattack.models.helpers.utils import load_cached_state_dict
from textattack.models.tokenizers import glove_tokenizer
from textattack.shared import utils
from miFunc_1 import (mask_string,
                      mlm_ummask,
                      masked_str_token,
                      get_mlm_um_sen,
                      gap_infer_acc_f1)
from classifier import euq_label,mean_mean_label,compute_score_difference_padding
from victim import predict_class,load_base_victim_model
from tqdm import tqdm
from transformers import (
                        AlbertForMaskedLM, AlbertTokenizer,
                        BertForMaskedLM, BertTokenizer, DistilBertModel,
                        DistilBertTokenizer, RobertaForMaskedLM,
                        RobertaTokenizer,
                        AutoModelForSequenceClassification,AutoTokenizer)


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_adv(path):

    def fetch_from_text(path):
        data=[]
        with open(path, "r") as f:  
            data_ = f.readlines()
        for dd in data_:
            data.append(dd[:-1])
        return data
    x_benigns=fetch_from_text(path+'_x_origs.txt')
    x_advs=fetch_from_text(path+'_x_advs.txt')
    return x_benigns,x_advs

def write_metric(x, file_name):
    with open(file_name, "w") as f:
        x_t = x.tolist()
        f.write(str(x_t))


def get_ben_adv_metric(path):
    def re_all(path):
        with open(path, "r") as f:  
            data_al= f.readlines()
        return data_al
    data_t= re_all(path)
    data = eval(data_t[0])
    return data


def write_um_sentences(x, file_name):
    with open(file_name, "w") as f:
       for idx in range(len(x)):
            f.write(str(x[idx] )+ '\n')

def get_um_sentences_list(path):
    data=[]
    with open(path, "r") as f:  
         data_ = f.readlines()
    for dd in data_:
        dd_t =eval(dd)
        data.append(dd_t)
    return data


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-tn", type=int, default=3)
    parser.add_argument("-n", type=int, default=500)
    parser.add_argument("-unm_model", default="roberta-base",
                        choices=['roberta-base', 'bert-base-uncased', 'albert-base-v2'])
    parser.add_argument("-hug", default=True)
    parser.add_argument("-victim", default="bert",
                        choices=['bert', 'albert', 'cnn', 'lstm']),
    parser.add_argument("-mfre", type=float, default=1)
    args = parser.parse_args()
    n=args.n
    candidate_token_num=args.tn
    huggingface_model=args.hug
    ma_fre=args.mfre
    

   #load masked language models.
    if args.unm_model=='roberta-base':
        tokenizer=RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
    elif args.unm_model=='bert-base-uncased':
        tokenizer=RobertaTokenizer.from_pretrained("bert-base-uncased")
        model = RobertaForMaskedLM.from_pretrained("bert-base-uncased")
    else:
        tokenizer=RobertaTokenizer.from_pretrained("albert-base-v2")
        model = RobertaForMaskedLM.from_pretrained("albert-base-v2")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)   
    model = model.to(device)
  
    #load detection file
    detection_file = 'advdata/ag-news/TextFooler_bert_x_advs.txt'
    
    dataset = detection_file.replace(".txt", "").split('/')[1]
    attack_method =detection_file.replace(".txt", "").split('/')[-1].split('_')[0]
    victim_model = detection_file.replace(".txt", "").split('/')[-1].split('_')[1]
    detect =detection_file.replace(".txt", "").split('_')[0]+'_'+victim_model
    
    # base_victim_model = ['lstm','cnn']

    #load victim model
    if huggingface_model:
        if args.victim  == 'albert':
            model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2")
            tokenizer_cls = AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
            # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-ag-news")
            # tokenizer_cls = AutoTokenizer.from_pretrained("textattack/albert-base-v2-ag-news")
            # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")
            # tokenizer_cls = AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
        else:
            # tokenizer_cls =AutoTokenizer.from_pretrained(f"textattack/{victim_model}-base-uncased-{dataset}")
            # model_cls = AutoModelForSequenceClassification.from_pretrained(f"textattack/{victim_model}-base-uncased-{dataset}")
            # tokenizer_cls =AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
            # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
            # tokenizer_cls =AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
            # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
            # tokenizer_cls =AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
            # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
            tokenizer_cls =AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
            model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
        if torch.cuda.device_count() > 1:
            model_cls = nn.DataParallel(model_cls)
        model_cls = model_cls.to(device)
    
    else:
        model_cls,tokenizer_cls = load_base_victim_model(victim_model)

    all_ben_masked_str=[]
    all_ben_masked_str_tokens=[]
    all_ben_um_sentence=[]
    all_ben_um_sentence_label=[]
    all_ben_euq_labels=[]
    all_ben_logits_out=[]

    # padding_socre_diff_f=[]
    x_benigns, x_advs = get_adv(detect)
    rand_index =np.random.choice(len(x_benigns),n)
    for idx in tqdm(range(n)):
        orig_str=x_benigns[rand_index[idx]]
        # masked_str=rand_mask_string(orig_str,tokenizer,ma_fre)
        masked_str=mask_string(orig_str,tokenizer)
        all_ben_masked_str.append(masked_str)
        
        masked_str=mask_string(orig_str,tokenizer)
        all_ben_masked_str.append(masked_str)
        
        masked_str_tokens = masked_str_token(masked_str,model,tokenizer)
        all_ben_masked_str_tokens.append(masked_str_tokens)
        
        um_sentence = get_mlm_um_sen(masked_str,masked_str_tokens,3)
        all_ben_um_sentence.append(um_sentence)
        
        um_sen_logits_out,um_sen_conf_socres,um_sen_label= predict_class(um_sentence,model_cls,tokenizer_cls,huggingface_model)
        all_ben_logits_out.append(um_sen_logits_out)
        all_ben_um_sentence_label.append(um_sen_label)
        
        #distinguishable score
        # _,orig_str_sorce,_= predict_class([orig_str],model_cls,tokenizer_cls)
        # padding_socre_diff=compute_score_difference_padding(orig_str_sorce,um_sen_conf_socres)
        # padding_socre_diff = padding_socre_diff[:,:1].unsqueeze(0)
        
        # padding_socre_diff_sig  = padding_socre_diff.cpu().numpy().flatten().tolist()
        # padding_socre_diff_f.append(padding_socre_diff_sig)
        
        euq_labels= euq_label(orig_str,um_sen_label,model_cls,tokenizer_cls,candidate_token_num)
        all_ben_euq_labels.append(euq_labels)

    all_adv_masked_str=[]
    all_adv_masked_str_tokens=[]
    all_adv_um_sentence=[]
    all_adv_um_sentence_label=[]
    all_adv_euq_labels=[]
    all_adv_logits_out=[]

    for idx in tqdm(range(n)):
        orig_str=x_advs[rand_index[idx]]
        # masked_str=rand_mask_string(orig_str,tokenizer,ma_fre)
        masked_str=mask_string(orig_str,tokenizer)
        all_adv_masked_str.append(masked_str)
        
        masked_str=mask_string(orig_str,tokenizer)
        all_adv_masked_str.append(masked_str)
        
        masked_str_tokens = masked_str_token(masked_str,model,tokenizer)
        all_adv_masked_str_tokens.append(masked_str_tokens)
        
        um_sentence = get_mlm_um_sen(masked_str,masked_str_tokens,3)
        all_adv_um_sentence.append(um_sentence)
        
        um_sen_logits_out,um_sen_conf_socres,um_sen_label= predict_class(um_sentence,model_cls,tokenizer_cls)
        all_adv_logits_out.append(um_sen_logits_out)
        all_adv_um_sentence_label.append(um_sen_label)
        
        #distinguishable score
        # _,orig_str_sorce,_= predict_class([orig_str],model_cls,tokenizer_cls)
        # padding_socre_diff=compute_score_difference_padding(orig_str_sorce,um_sen_conf_socres)
        # padding_socre_diff = padding_socre_diff[:,:1].unsqueeze(0)

        
        # padding_socre_diff_sig  = padding_socre_diff.cpu().numpy().flatten().tolist()
        # padding_socre_diff_f.append(padding_socre_diff_sig)
        
        euq_labels= euq_label(orig_str,um_sen_label,model_cls,tokenizer_cls,candidate_token_num)
        all_adv_euq_labels.append(euq_labels)
    
    ben_metric=mean_mean_label(all_ben_euq_labels)   
    adv_metric=mean_mean_label(all_adv_euq_labels)
    gap_a,acc,gap_f,f1=gap_infer_acc_f1(ben_metric, adv_metric)
    print('unmask model:',args.unm_model)
    print('candidate_token_num:',candidate_token_num)
    print("mask frequency is :",ma_fre)
    print('test config:',detection_file)
    print('********')
    print('gap_a:',gap_a,"ACC:",acc,"gap_f:",gap_f,'F1:',f1)