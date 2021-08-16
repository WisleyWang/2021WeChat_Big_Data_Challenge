#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import pandas as pd
from torch import nn
import dgl.function as fn
import torch.nn.functional as F
import gc
import random
from tqdm import tqdm
from sklearn.metrics import auc,roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models.deepfm import FM,DNN
from deepctr_torch.layers  import CIN,InteractingLayer,CrossNet,CrossNetMix
from deepctr_torch.models.basemodel import *
from collections import defaultdict
from torch.optim import Optimizer
import torchtext
import os
import pickle
import warnings
from model.model4 import *
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
SEED=4096
setup_seed(SEED)

warnings.filterwarnings("ignore")
ROOT_PATH='../../data/'
ACTION_LIST = ["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']#
PREDICT_LIST=["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']
##  加载数据----------------------------
feed_emb=np.load(ROOT_PATH+'tmp/feed_emb.npy')
userid2nid=pickle.load(open(ROOT_PATH+'tmp/userid2nid.pkl','rb'))
feedid2nid=pickle.load(open(ROOT_PATH+'tmp/feedid2nid.pkl','rb'))
item_texts=pickle.load(open(ROOT_PATH+'tmp/item_texts.pkl','rb'))
feed_data=pickle.load(open(ROOT_PATH+'tmp/feed_data.pkl','rb'))
user_data=pickle.load(open(ROOT_PATH+'tmp/user_data.pkl','rb'))
graph_emb=np.concatenate([np.load(ROOT_PATH+'tmp/grap_allembedding64+day_hs2.npy'),np.load(ROOT_PATH+'tmp/grap_allembedding64+day_sg2.npy')],axis=1)
graph_user=np.load(ROOT_PATH+'tmp/grap_userid64+day_hs2.npy')
# np.concatenate([np.load(ROOT_PATH+'tmp/grap_allembedding32_sg2.npy'),np.load(ROOT_PATH+'tmp/grap_allembedding32_hs2.npy')],axis=1)
#--------------------------------------------------------
tokenize = lambda x: x.split(' ')
fields = {}
examples = []
for key, texts in item_texts.items():
    if  key in ['ocr','asr','description']:
        fields[key] = torchtext.data.Field(include_lengths=True, lower=True,tokenize=tokenize, batch_first=True, fix_length=64)
    else:
        fields[key] = torchtext.data.Field(include_lengths=True, lower=True,tokenize=tokenize, batch_first=True, fix_length=5)
for i in range(len(item_texts['ocr'])):
    example = torchtext.data.Example.fromlist(
        [item_texts[key][i] for key in item_texts.keys()],
        [(key, fields[key]) for key in item_texts.keys()])  #( [feat1,feat2], [(key1,field1),(key2,field2)] )
    examples.append(example)
textset = torchtext.data.Dataset(examples, fields)
for key, field in fields.items():
    field.build_vocab(getattr(textset, key))
for field_name, field in textset.fields.items():
    examples = [getattr(textset[i], field_name) for i in range(len(textset.examples))]

    tokens, lengths = field.process(examples)

    if not field.batch_first:
        tokens = tokens.t()
    # 给feed +上文本向量
    feed_data[field_name] = tokens
# ---------------------------------------
#加载df 和训练集
df=pd.read_pickle(ROOT_PATH+'tmp/ratings_feat_df.pkl')
# df=reduce_mem(df)

print('-------------------开始训练---------------------------------')
max_day=15
train_ratings=df#[(df.date_<max_day)]
del df
gc.collect()
# test_a=df[df.date_==15]
batch_size=4096*3
epochs=2
#-------------------------------------------

model = Model4(user_data,feed_data,textset
             ,feed_embed=feed_emb,graph_emb=graph_emb,user_graph=graph_user) #in_features, hidden_features, out_features, rel_names
# model = nn.DataParallel(model)
model = model.cuda()
for f,d in user_data.items():
    user_data[f]=d.cuda()
    
for f,d in feed_data.items():
    feed_data[f]=d.cuda()
# model=model.to(torch.device('cuda'))
# model = nn.DataParallel(model)
#------------------------------
train_steps = int(len(train_ratings) * epochs / batch_size) + 1
optimizer, scheduler = build_optimizer(model, train_steps, learning_rate=2.5e-3)
all_pred=[]
src=torch.from_numpy(train_ratings['userid'].apply(lambda x: userid2nid[x]).values).long()
dst=torch.from_numpy(train_ratings['feedid'].apply(lambda x: feedid2nid[x]).values).long()
labels=torch.from_numpy(train_ratings[PREDICT_LIST].values).float()

criti=nn.BCEWithLogitsLoss()
n_pos=len(train_ratings)
batch_index=np.arange(n_pos) # 生成正样本的index
for epoch in range(epochs):
    print('epoch: ----%d--'%epoch)
    random.shuffle(batch_index) 
    epoch_loss=0
    model.train()
    for ind in tqdm(range(0,n_pos//batch_size+1)):
        batch=batch_index[ind*batch_size:(ind+1)*batch_size]
        batch_src=src[batch]
        batch_dst=dst[batch]
        logits = model(batch_src,batch_dst)
        batch_label=labels[batch].cuda()
        loss=criti(logits[0][:,0],batch_label[:,0])*0.8+criti(logits[1][:,0],batch_label[:,1])*0.8+\
        criti(logits[2][:,0],batch_label[:,2])*0.4+criti(logits[3][:,0],batch_label[:,3])*0.4+\
        criti(logits[4][:,0],batch_label[:,4])*0.3+criti(logits[5][:,0],batch_label[:,5])*0.3+criti(logits[6][:,0],batch_label[:,6])*0.3
        epoch_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        if ind%1000==0:
            print('binary loss:',loss.item())
            batch_label=batch_label.cpu().numpy()
            pred=torch.cat(logits,axis=-1).sigmoid().detach().cpu().numpy()
#             pred=logits.sigmoid().detach().cpu().numpy()
            for ii,aa in enumerate(PREDICT_LIST):
                try:
                    print('train %s auc:'%aa,roc_auc_score(batch_label[:,ii],pred[:,ii]))
                except:
                    continue
    print('epoch %d  loss: %f '%(epoch,epoch_loss/(len(batch_index)//batch_size+1)))
#  保存模型
torch.save(model, ROOT_PATH+'/model/model4.pth')
#---------------------------------------------------
# test_pred=test_pred_func(model)
# test_a[PREDICT_LIST]=test_pred[:,:len(PREDICT_LIST)]
# sub=test_a[['userid','feedid']+PREDICT_LIST]
# # for i in range(len(PREDICT_LIST)):
# #     sub[PREDICT_LIST[i]]=test_pred[:,i]

# sub.to_csv('./upload/deep_v2_v1.csv',index=False)