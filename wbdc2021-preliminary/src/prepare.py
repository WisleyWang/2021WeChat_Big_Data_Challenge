#!/usr/bin/env python
# coding: utf-8
'''
该文件生成聚类数据:../data/tmp/feed_4096_knn.npy
graph embedding:
../data/tmp/grap_embedding32_feedid_b1.npy
../data/tmp/grap_embedding64_feedid_b1.npy
'''
import cuml
import numpy as np
import torch
import dgl
import pandas as pd
import dgl.nn as dglnn
from torch import nn
import dgl.function as fn
import torch.nn.functional as F
import gc
import os
import random
from tqdm import tqdm
from sklearn.metrics import auc,roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models.deepfm import FM,DNN
from deepctr_torch.models.basemodel import *
from collections import defaultdict
from torch.optim import Optimizer
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence
import logging
import gensim
import os
from prepareFeatureLightgbm import prepareFeature


# In[2]:
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))

DATA_PATH=os.path.join(BASE_DIR, '../data/wedata/wechat_algo_data1/')
TMP_PATH=os.path.join(BASE_DIR, '../data/tmp/')
SUBMMIT_PATH=os.path.join(BASE_DIR, '../data/submission/')
MODEL_PATH=os.path.join(BASE_DIR, '../data/model/')

ratings=pd.read_csv(os.path.join(DATA_PATH,'user_action.csv'))
test_a=pd.read_csv(os.path.join(DATA_PATH,'test_a.csv'))
test_b=pd.read_csv(os.path.join(DATA_PATH,'test_b.csv'))

feed_info=pd.read_csv(os.path.join(DATA_PATH,'feed_info.csv'))
feed_embedding=pd.read_csv(os.path.join(DATA_PATH,'feed_embeddings.csv'))

ACTION_LIST = ["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']#
PREDICT_LIST=["read_comment","like", "click_avatar", "forward"]

user_info=ratings.drop_duplicates('userid','first')[['userid','device']]
# 
print(ratings.shape)
ratings=ratings.drop_duplicates(['userid','feedid'],'last')
print(ratings.shape)

## 建立 user _item 的对应表
userid_list=ratings.userid.unique().tolist()
userid2nid=dict(zip(userid_list,range(len(userid_list))))
num_user=len(userid_list)
print('user nums',num_user)

feedid_list=feed_info.feedid.unique().tolist()
num_feed=len(feedid_list)
print('feed nums',num_feed)

num_node_all=num_user+num_feed
print('all node nums',num_node_all)

feedid2nid=dict(zip(feedid_list,range(num_feed)))

# 聚类 保存
feed_embedding['node']=feed_embedding['feedid'].apply(lambda x:feedid2nid[x])
feed_matrix=np.vstack(feed_embedding['feed_embedding'].apply(lambda x:np.array(list(map(float,x.strip().split(' '))))).values)

feed_emb=np.zeros(feed_matrix.shape)
indexs=feed_embedding['node'].values
for i in range(feed_matrix.shape[0]):
    feed_emb[indexs[i]]=feed_matrix[i]
del feed_embedding,feed_matrix

kmean=cuml.cluster.KMeans(n_clusters=4096)
pred=kmean.fit_transform(feed_emb)
np.save(os.path.join(TMP_PATH,'feed_4096_knn.npy'),kmean.labels_)
print("labels:")
print(kmean.labels_)
print("cluster_centers:")
print(kmean.cluster_centers_)
#------------------------------------------------------

# 训练 graph emb --- w2v
df=pd.concat([ratings[['userid','feedid']],test_a[['userid','feedid']],test_b[['userid','feedid']]],axis=0)
df['feedid_node']=df['feedid'].apply(lambda x:feedid2nid[x])
tmp=df.groupby('userid')['feedid_node'].apply(lambda x : ' '.join([str(i) for i in x]))


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 64 dim--------------------- 训练 保存 dim=64的feedid emb
w2v=Word2Vec(tmp.apply(lambda x: x.split(' ')).tolist(),size=64, window=12, iter=40, min_count=2,
                     sg=1, sample=0.0005, workers=4 , seed=1018)
#window12 6 16
w2v.wv.save_word2vec_format(os.path.join(TMP_PATH,'w2v_feedid_64_b1.txt'))
feed_emb=np.zeros((len(feed_info),64))
for i in range(len(feed_info)):
    if str(i) not in w2v.wv.vocab:
        continue
    feed_emb[i]=w2v[str(i)]
np.save(os.path.join(TMP_PATH,'grap_embedding64_feedid_b1.npy'),feed_emb)

# 32 dim------------------------训练 保存 dim=32的feedid emb
w2v=Word2Vec(tmp.apply(lambda x: x.split(' ')).tolist(),size=32, window=12, iter=40, min_count=2,
                     sg=1, sample=0.0005, workers=4 , seed=1018)
#window12 6 16
w2v.wv.save_word2vec_format(os.path.join(TMP_PATH,'w2v_feedid_32_b1.txt'))
feed_emb=np.zeros((len(feed_info),32))
for i in range(len(feed_info)):
    if str(i) not in w2v.wv.vocab:
        continue
    feed_emb[i]=w2v[str(i)]
np.save(os.path.join(TMP_PATH,'grap_embedding32_feedid_b1.npy'),feed_emb)
