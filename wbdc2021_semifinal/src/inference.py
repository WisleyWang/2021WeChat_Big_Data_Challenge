#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import pandas as pd
from torch import nn
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
import argparse
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#----添加model到系统环境-----------
import sys
sys.path.append('./src/train/')
#------加载模型1------------------
from train.model.model1 import * # 转入模型
##  处理数据---------------------------------------------
ROOT_PATH='./data/'
PREDICT_LIST=["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']
parser=argparse.ArgumentParser(description='----inference--')
parser.add_argument('submit', type=str, help='测试集地址')
#     parser.add_argument('--out-path',type=str,default='./data/submission/')
args=parser.parse_args()
    
test_a=pd.read_csv(args.submit)
print('----开始推断1---------')
PREDICT_LIST=["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']
# test_a=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/test_a.csv')
feed_info=pd.read_pickle(ROOT_PATH+'tmp/cat_feed_info.pkl')
userid2nid=pickle.load(open(ROOT_PATH+'tmp/userid2nid.pkl','rb'))
feedid2nid=pickle.load(open(ROOT_PATH+'tmp/feedid2nid.pkl','rb'))
test_a=test_a.merge(feed_info[['feedid', 'authorid', 'videoplayseconds','manual_keyword_id1','manual_tag_id1']], on='feedid', how='left')
test_a['date_']=15 # 由于 复赛数据B榜不可见 这里就不放入a榜数据使用
max_day=15
for stat_cols in tqdm([['userid'],['feedid'],['authorid'], ['userid', 'authorid'],['userid', 'manual_tag_id1'],
        ['userid','manual_keyword_id1']]):
    f = '_'.join(stat_cols)
#     tmp.to_pickle('./tmp/{}_feat_{}.pkl'.format(target_day,'_'.join(stat_cols)))
    tmp=pd.read_pickle(ROOT_PATH+'tmp/{}_feat_{}.pkl'.format(15,'_'.join(stat_cols)))
    tmp=reduce_mem(tmp)
    test_a = test_a.merge(tmp, on=stat_cols + ['date_'], how='left')
    mean_tmp=pickle.load(open(ROOT_PATH+'tmp/{}_feat_mean.pkl'.format('_'.join(stat_cols)),'rb'))
    for kk,vv in mean_tmp.items():
        test_a[kk]=test_a[kk].fillna(vv) # 填充均值
test_a=reduce_mem(test_a)
gc.collect()
# 加载特征list-------------------------
feat=pickle.load(open(ROOT_PATH+'tmp/feat_list.pkl','rb'))
#--------------归一化分布-----------
normolizer_dict=pickle.load(open(ROOT_PATH+'tmp/normolizer_dict.pkl','rb'))
for f in tqdm(feat):
    tmp=test_a[f].values.astype('float16').clip(-1,1e8)
    tmp_max=normolizer_dict[f+'_max'] # 这里 或许我得保留均值和方差
    tmp_min=normolizer_dict[f+'_min']
    test_a[f]=((tmp-tmp_min)/tmp_max).astype('float16')
    
src=torch.from_numpy(test_a['userid'].apply(lambda x: userid2nid[x]).values).long()
dst=torch.from_numpy(test_a['feedid'].apply(lambda x: feedid2nid[x]).values).long()
test_dense=torch.from_numpy(test_a[feat].values).float()
# ------------trans-------------------------    
hist_seq=torch.from_numpy(np.load(ROOT_PATH+'tmp/hist_list1.npy')).long()
hist_id=torch.from_numpy((test_a['date_'].values-1)*len(userid2nid)).long()+src 
hist_seq2=torch.from_numpy(np.load(ROOT_PATH+'tmp/hist_list1.npy')).long()
#-----------------trans1-------------------------------------------
from train.model.trans1 import * # 转入模型
model = torch.load(ROOT_PATH+'model/trans.pth')
test_pred=test_pred_func(model,test_a,src,dst,hist_id,hist_seq,batch_size=4096*3)
test_a[PREDICT_LIST]=test_pred
subt1=test_a[['userid','feedid']+PREDICT_LIST]
#---------------trains1-hist2----------------------
model = torch.load(ROOT_PATH+'model/trans1_hist2.pth')
test_pred=test_pred_func(model,test_a,src,dst,hist_id,hist_seq2,batch_size=4096*3)
test_a[PREDICT_LIST]=test_pred
subt3=test_a[['userid','feedid']+PREDICT_LIST]
#-------------------reansv2-----------
from train.model.trans1 import * # 转入模型
model = torch.load(ROOT_PATH+'model/transv2.pth')
test_pred=test_pred_func(model,test_a,src,dst,hist_id,hist_seq,batch_size=4096*3)
test_a[PREDICT_LIST]=test_pred
subt2=test_a[['userid','feedid']+PREDICT_LIST]
#----------modelv4----------------------------
from train.model.model4 import *
model = torch.load(ROOT_PATH+'model/model4.pth')
test_pred=test_pred_func(model,test_a,src,dst,batch_size=4096*3)
test_a[PREDICT_LIST]=test_pred
sub4=test_a[['userid','feedid']+PREDICT_LIST]

#-----------------------------------------
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from train.model.model1 import * # 转入模型
model = torch.load(ROOT_PATH+'model/model1.pth')
#----------------inference-------------------------------------------------
test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3)
for i,f in enumerate(PREDICT_LIST):
    test_a[f]=test_pred[:,i]
sub1=test_a[['userid','feedid']+PREDICT_LIST]
#---
model = torch.load(ROOT_PATH+'model/model1-1.pth')
test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3)
for i,f in enumerate(PREDICT_LIST):
    test_a[f]=test_pred[:,i]
sub1_1=test_a[['userid','feedid']+PREDICT_LIST]


 #------加载模型2------------------
from train.model.model2 import * # 转入模型
model = torch.load(ROOT_PATH+'model/model2.pth')
#----------------inference-------------------------------------------------
test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3)
for i,f in enumerate(PREDICT_LIST):
    test_a[f]=test_pred[:,i]
sub2=test_a[['userid','feedid']+PREDICT_LIST]

##------加载模型3------------------
from train.model.model3 import * # 转入模型
model = torch.load(ROOT_PATH+'model/model3.pth')
#----------------inference-------------------------------------------------
test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3)
for i,f in enumerate(PREDICT_LIST):
    test_a[f]=test_pred[:,i]
sub3=test_a[['userid','feedid']+PREDICT_LIST]
#---------------------------------

#---------融合---------------------
sub=sub1.copy()
sub[PREDICT_LIST]=(subt1[PREDICT_LIST]*0.5+subt2[PREDICT_LIST]*0.25+subt3[PREDICT_LIST]*0.25)/2+\
(sub1[PREDICT_LIST]*0.3+sub2[PREDICT_LIST]*0.2+sub3[PREDICT_LIST]*0.2+sub4[PREDICT_LIST]*0.2+sub1_1[PREDICT_LIST]*0.1)/2

sub.to_csv(os.path.join(ROOT_PATH+'submission','result.csv'),index=False)

