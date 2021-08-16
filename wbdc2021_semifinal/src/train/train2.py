from torch import nn
import pandas as pd
import numpy as np
import pickle
import torchtext
from torch.optim import Optimizer
import torch
from deepctr_torch.models.deepfm import FM,DNN
from deepctr_torch.layers  import CIN,InteractingLayer,CrossNet,CrossNetMix,AttentionSequencePoolingLayer
from deepctr_torch.models.basemodel import *
from tqdm import tqdm
import random
import gc
from collections import defaultdict
from sklearn.metrics import auc,roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
from model.trans1 import *
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
SEED=1988
setup_seed(SEED)
ROOT_PATH='../../data/'
user_data=pickle.load(open(ROOT_PATH+'tmp/user_data.pkl','rb'))

feed_data=pickle.load(open(ROOT_PATH+'tmp/feed_data.pkl','rb'))

item_texts=pickle.load(open(ROOT_PATH+'tmp/item_texts.pkl','rb'))
feed_emb=np.load(ROOT_PATH+'tmp/feed_emb.npy')
userid2nid=pickle.load(open(ROOT_PATH+'tmp/userid2nid.pkl','rb'))
feedid2nid=pickle.load(open(ROOT_PATH+'tmp/feedid2nid.pkl','rb'))
graph_emb=np.concatenate([np.load(ROOT_PATH+'tmp/grap_allembedding32_sg2.npy'),np.load(ROOT_PATH+'tmp/grap_allembedding32_hs2.npy')],axis=1)
ratings=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/user_action.csv')

tokenize = lambda x: x.split(' ')
fields = {}
examples = []
for key, texts in item_texts.items():
    if  key in ['ocr','asr','description']:
        fields[key] = torchtext.data.Field(include_lengths=True, lower=True,tokenize=tokenize, batch_first=True, fix_length=64)
    else:
        fields[key] = torchtext.data.Field(include_lengths=True, lower=True,tokenize=tokenize, batch_first=True, fix_length=5)
    
for i in range(len(feedid2nid)):
    example = torchtext.data.Example.fromlist(
        [item_texts[key][i] for key in item_texts.keys()],
        [(key, fields[key]) for key in item_texts.keys()])  #( [feat1,feat2], [(key1,field1),(key2,field2)] )
    examples.append(example)
textset = torchtext.data.Dataset(examples, fields)
for key, field in fields.items():
    field.build_vocab(getattr(textset, key))
for field_name, field in textset.fields.items():
    examples = [getattr(textset[i], field_name) for i in range(len(feedid2nid))]
    tokens, lengths = field.process(examples)
    if not field.batch_first:
        tokens = tokens.t()
    # 给feed +上文本向量
    feed_data[field_name] = tokens
    
PREDICT_LIST=["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']
max_day=15
train_ratings=ratings[(ratings.date_<max_day)]
# val_ratings=ratings[ratings.date_==max_day]
# del ratings
gc.collect()

src=torch.from_numpy(train_ratings['userid'].apply(lambda x: userid2nid[x]).values).long()
dst=torch.from_numpy(train_ratings['feedid'].apply(lambda x: feedid2nid[x]).values).long()
hist_id=torch.from_numpy((train_ratings['date_'].values-1)*len(userid2nid)).long()+src
labels=torch.from_numpy(train_ratings[PREDICT_LIST].values).float()
hist_seq=torch.from_numpy(np.load(ROOT_PATH+'tmp/hist_list2.npy')).long()
batch_size=4096*2
epochs=2
device=torch.device('cuda:0')
for f,d in user_data.items():
    user_data[f]=d.to(device)
for f,d in feed_data.items():
    feed_data[f]=d.to(device)
model = Transmodel(user_data,feed_data,textset=textset
             ,feed_embed=feed_emb,graph_emb=graph_emb,device=device)
model=model.to(device)
train_steps = int(len(train_ratings) * epochs / batch_size) + 1
optimizer, scheduler = build_optimizer(model, train_steps, learning_rate=2e-2)
all_pred=[]

criti=nn.BCEWithLogitsLoss()
reg_criti=nn.MSELoss()
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
        batch_hist=hist_seq[hist_id[batch]]
#         print(batch_src)
        logits =model(batch_src,batch_dst,batch_hist[:,:-1],batch_hist[:,-1:].to(device))
        batch_label=labels[batch].to(device)
        loss=criti(logits[0][:,0],batch_label[:,0])*0.8+criti(logits[1][:,0],batch_label[:,1])*0.7+\
        criti(logits[2][:,0],batch_label[:,2])*0.6+criti(logits[3][:,0],batch_label[:,3])*0.6+\
        criti(logits[4][:,0],batch_label[:,4])*0.6+criti(logits[5][:,0],batch_label[:,5])*0.6+criti(logits[6][:,0],batch_label[:,6])*0.6
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
#     n_evaluate_nn(model,val_df=val_ratings,action_list=PREDICT_LIST,batch_size=2048,device=device)
    
torch.save(model, ROOT_PATH+'/model/trans1_hist2.pth')

