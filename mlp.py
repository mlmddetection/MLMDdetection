import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Text(Dataset):
    def __init__(self, x , y):
        self.y = y
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = torch.tensor(self.x[idx].astype('float32')).to(device)
        y = torch.tensor(self.y[idx].astype('float32')).unsqueeze(0).to(device)
        return data, y


#load distinguishable score. 
# natural means orginal distinguishable score and sorted denotes that applying rank operation to distinguishable score.
ttrain_path ='data_train/ag-news/ag-news_TextFooler_bert_x_natural.npy'
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
if_natural=False

if if_natural:
    train_ds = Text(x_ttest_n[:train_num],y_ttest[:train_num])
    x_test =x_ttest_n[train_num:]
else:
    train_ds = Text(x_ttest_s[:train_num],y_ttest[:train_num])
    x_test =x_ttest_s[train_num:]
    
y_test =y_ttest[train_num:]
train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True)

class BasicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim  = output_dim

        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
    
basic_classifier = BasicModel(input_dim=1536*1, hidden_dim=400, output_dim=1).to(device)
c = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(basic_classifier.parameters(), lr=0.001)

train_loss_history = []
val_acc_history = []

iter_per_epoch = len(train_loader)
num_epochs =50
initial_epoch = 1
log_nth = 2
storing_frequency = 15

for epoch in range(initial_epoch, initial_epoch+num_epochs):
    basic_classifier.train()
    epoch_losses = []
    for i, (data, y_label) in enumerate(train_loader):
      optimizer.zero_grad()
      out = basic_classifier(data)
      loss = c(out, y_label)
      epoch_losses.append(loss.item())
      loss.backward()
      optimizer.step()

from sklearn.metrics import classification_report, confusion_matrix

nn_pred = basic_classifier(torch.tensor(x_test.astype('float32')).to(device))
nn_pred = nn_pred.flatten().detach().cpu().numpy().round()
np.sum(nn_pred==y_test)/len(y_test)
    
print(classification_report(y_test, nn_pred, digits=5))
print(confusion_matrix(y_test, nn_pred))
tn,fp,fn,tp= confusion_matrix(y_ttest[train_num:], nn_pred).ravel()
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