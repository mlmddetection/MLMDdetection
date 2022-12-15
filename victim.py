import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
softmax = nn.Softmax(dim=1)
import importlib
#load base victim models
import time
import re
def predict_class(um_sentence_list,model_cls,tokenizer_cls,huggingface_model):
    logits = []
    scores = []
    if len(um_sentence_list) > 100:   
        batches = [um_sentence_list[i:i + 100] for i in range(0, len(um_sentence_list), 100)]
        for b in batches:
            if huggingface_model: # Use hugging face predictions
                batch = tokenizer_cls(b, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    logits_t = model_cls(**batch).logits
                    logits.append(logits_t.cpu().numpy())

                    score_t = softmax(logits_t)
                    scores.append(score_t)

            else:
                logits_t = model_cls(b)
                logits.append(logits_t)

                score_t = softmax(torch.tensor(logits_t))
                scores.append(score_t)
      
        scores= torch.cat(scores)
    else:
        if huggingface_model:
            batch = tokenizer_cls(um_sentence_list, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                logits_t =model_cls(**batch).logits
                logits.append(logits_t.cpu().numpy())

                scores= softmax(logits_t)
        else:
            logits_t = model_cls(um_sentence_list)
            logits.append(logits_t)


            scores= softmax(torch.tensor(logits_t))
    logits_flatten =[i for k in logits for i in k]
    label = [np.argmax(i) for i in logits_flatten]
    return logits,scores,label

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
