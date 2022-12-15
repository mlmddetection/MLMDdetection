

import importlib
import time
# IMDb or SST-2
def get_advs(address):
    all_data = []
    with open(address, "r") as f:
        data = f.readlines()
    for dd in data:
        all_data.append(dd[:-1])
    x_advs = []
    x_origs = []
    y_advs = []
    y_origs = []

    import re
    pattern_p = re.compile(r'0\s')
    pattern_n = re.compile(r'1\s')
    for idx in range(len(all_data)):
        if len(all_data[idx]) == 0:
            continue
        if all_data[idx][0] == '-':
            ret1 = pattern_p.search(all_data[idx + 1])
            ret2 = pattern_n.search(all_data[idx + 1])
            proce = re.compile(r'<br\s/>')
            if (ret1 is not None) and (ret2 is not None):
                all_data[idx + 3] =all_data[idx + 3].replace('[', '')
                all_data[idx + 3] =all_data[idx + 3].replace(']', '')
                all_data[idx + 3] =proce.sub('',all_data[idx + 3])
                
                all_data[idx + 5] =all_data[idx + 5].replace('[', '')
                all_data[idx + 5] =all_data[idx + 5].replace(']', '')
                all_data[idx + 5] =proce.sub('',all_data[idx + 5])
                
                x_origs.append(all_data[idx + 3])
                x_advs.append(all_data[idx + 5])
        
                if ret1 is not None:
                    ret1 = ret1.regs[0]
                    if ret1[0] == 2:
                        y_origs.append(0)
                        y_advs.append(1)
                if ret2 is not None:
                    ret2 = ret2.regs[0]
                    if ret2[0] == 2:
                        y_origs.append(1)
                        y_advs.append(0)
                    
    return x_advs, x_origs, y_advs, y_origs

#AG-NEWS
# def get_advs(address):
#     all_data = []
#     with open(address, "r") as f:
#         data = f.readlines()
#     for dd in data:
#         all_data.append(dd[:-1])
#     x_advs = []
#     x_origs = []
#     y_advs = []
#     y_origs = []
#     import re
#     proce = re.compile(r'<br\s/>')
#     # pattern_p = re.compile(r'0\s')#\[\[[0-9]\s*\(([0-9]*)\%\)\]\]   0\s
#     pattern_p = re.compile(r'0\s')
#     pattern_n1 = re.compile(r'1\s')#\[\[.*?\(([0-9]*)\%\)\]\]    1\s
#     pattern_n2 = re.compile(r'2\s')
#     pattern_n3 = re.compile(r'3\s')
#     for idx in range(len(all_data)):
#         if len(all_data[idx]) == 0:
#             continue
#         if all_data[idx][0] == '-':
#             ret1 = pattern_p.search(all_data[idx + 1])
#             ret2 = pattern_n1.search(all_data[idx + 1])
#             ret3 = pattern_n2.search(all_data[idx + 1])
#             ret4 = pattern_n3.search(all_data[idx + 1])
#             if ((ret1 is not None) and (ret2 is not None)) or ((ret1 is not None) and (ret3 is not None)) or((ret1 is not None) and (ret4 is not None))or((ret2 is not None) and (ret3 is not None))or((ret2 is not None) and (ret4 is not None))or((ret3 is not None) and (ret4 is not None)):       
#                 all_data[idx + 3] =all_data[idx + 3].replace('[', '')
#                 all_data[idx + 3] =all_data[idx + 3].replace(']', '')
#                 all_data[idx + 3] =proce.sub('',all_data[idx + 3])
#                 x_origs.append(all_data[idx + 3])
#                 all_data[idx + 5] =all_data[idx + 5].replace('[', '')
#                 all_data[idx + 5] =all_data[idx + 5].replace(']', '')
#                 all_data[idx + 5] =proce.sub('',all_data[idx + 5])
#                 x_advs.append(all_data[idx + 5])
#             if ret1 is not None:
#                 ret1 = ret1.regs[0]
#                 if (ret1[0] == 2)and(ret2 is not None):
#                     y_origs.append(0)
#                     y_advs.append(1)
#                 if (ret1[0] == 2)and(ret3 is not None):
#                     y_origs.append(0)
#                     y_advs.append(2)
#                 if (ret1[0] == 2)and(ret4 is not None):
#                     y_origs.append(0)
#                     y_advs.append(3)                      
#             if ret2 is not None:
#                 ret2 = ret2.regs[0]
#                 if (ret2[0] == 2)and (ret1 is not None):
#                     y_origs.append(1)
#                     y_advs.append(0)
#                 if (ret2[0] == 2)and(ret3 is not None):
#                     y_origs.append(1)
#                     y_advs.append(2)
#                 if (ret2[0] == 2)and(ret4 is not None):
#                     y_origs.append(1)
#                     y_advs.append(3)
#             if ret3 is not None:
#                 ret3 = ret3.regs[0]
#                 if (ret3[0] == 2)and (ret1 is not None):
#                     y_origs.append(2)
#                     y_advs.append(0)
#                 if (ret3[0] == 2)and(ret2 is not None):
#                     y_origs.append(2)
#                     y_advs.append(1)
#                 if (ret3[0] == 2)and(ret4 is not None):
#                     y_origs.append(2)
#                     y_advs.append(3)
#             if ret4 is not None:
#                 ret4 = ret4.regs[0]
#                 if (ret4[0] == 2)and (ret1 is not None):
#                     y_origs.append(3)
#                     y_advs.append(0)
#                 if (ret4[0] == 2)and(ret2 is not None):
#                     y_origs.append(3)
#                     y_advs.append(1)
#                 if (ret4[0] == 2)and(ret3 is not None):
#                     y_origs.append(3)
#                     y_advs.append(2)  
#     return x_advs, x_origs, y_advs, y_origs

def predict_class(x_list,model_cls,tokenizer_cls):
    logits_out=[]
   
    batches =[x_list[i:i+100] for i in range(0,len(x_list),100)]
    for b in batches:
        if huggingface_model:
            batch = tokenizer_cls(b, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                logits_out_t= model_cls(**batch).logits.cpu().numpy()
                logits_out.append(logits_out_t)
        else:
            logits_out_t= model_cls(b)
            logits_out.append(logits_out_t)
    logits_out_flatten =[i for k in logits_out for i in k]
    label = [np.argmax(i) for i in logits_out_flatten]
    return logits_out,label


def load_base_victim_model(victim_model):
    
    def load_module_from_file(file_path):
        """Uses ``importlib`` to dynamically open a file and load an object from
        it."""
        temp_module_name = f"temp_{time.time()}"

        spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    m = load_module_from_file(f'{victim_model}.py')
    model = getattr(m, 'model')  
    # print("model",model)
    return model, None

def write_txt(x, file_name):
    with open(file_name, "w") as f:
        for xi in x:
            f.write(xi + '\n') 
              
def get_info(x_advs, x_origs, y_advs, y_origs): 
    x_advs_final=[]
    x_origs_final=[]
    y_advs_pred_final=[]
    y_origs_pred_final=[]
    y_advs_final=[]
    y_origs_final=[]
    for idx in range(len(x_origs)):
        if int(y_origs_pred[idx])!=int(y_advs_pred[idx]):
            x_origs_final.append(x_origs[idx])
            y_origs_final.append(y_origs[idx])
            y_origs_pred_final.append(y_origs_pred[idx])
        
            x_advs_final.append(x_advs[idx])
            y_advs_final.append(y_advs[idx])
            y_advs_pred_final.append(y_advs_pred[idx])
    return x_origs_final,y_origs_final,y_origs_pred_final,x_advs_final,y_advs_final,y_advs_pred_final





            
import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

if __name__=='__main__':       
    #Obtain test examples and theri labels.
    advs_file = "attack_log/imdb/PWWS_bert_imdb_log.txt"
    attack_method =advs_file.replace(".txt", "").split('/')[-1].split('_')[0]
    victim_model = advs_file.replace(".txt", "").split('/')[-1].split('_')[1]
    dataset = advs_file.replace(".txt", "").split('/')[-1].split('_')[2]
    base_victim_model = ['lstm','cnn']

    x_advs, x_origs, y_advs, y_origs =get_advs(advs_file)


    if victim_model in base_victim_model:
        huggingface_model=False
        model_cls,tokenizer_cls = load_base_victim_model(victim_model)
    else:
        huggingface_model=True
        # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")
        # tokenizer_cls = AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")

        # tokenizer_cls =AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")

        # tokenizer_cls =AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
        # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")

        # model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-ag-news")
        # tokenizer_cls = AutoTokenizer.from_pretrained("textattack/albert-base-v2-ag-news")
        
        tokenizer_cls =AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
        model_cls = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")

        if torch.cuda.device_count() > 1:
            model_cls = nn.DataParallel(model_cls)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_cls = model_cls.to(device)

    y_advs_logits,y_advs_pred=predict_class(x_advs,model_cls,tokenizer_cls)
    y_origs_logits,y_origs_pred=predict_class(x_origs,model_cls,tokenizer_cls)

    x_origs_final,y_origs_final,y_origs_pred_final,x_advs_final,y_advs_final,y_advs_pred_final=get_info(x_advs, x_origs, y_advs, y_origs)

    #data for MLMD.
    dataset_name = "imdb"
    attack_name = attack_method
    victim = victim_model

    write_txt(x_origs_final, 'advdata/' + dataset_name + '/' + attack_name+ '_' + victim +'c'+'_x_origs.txt')
    write_txt(x_advs_final, 'advdata/' + dataset_name + '/' + attack_name + '_'+ victim + 'c'+'_x_advs.txt')
    np.save('advdata/' + dataset_name + '/' + attack_name+ '_' + victim +'c'+'_y_origs',y_origs_pred_final)
    np.save('advdata/' + dataset_name + '/' + attack_name+ '_' + victim +'c'+'_y_advs',y_advs_pred_final)     
        
    

