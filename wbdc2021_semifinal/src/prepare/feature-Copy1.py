import numpy as np
import torch
import pandas as pd
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
from deepctr_torch.layers  import CIN,InteractingLayer,CrossNet,CrossNetMix
from deepctr_torch.models.basemodel import *
from collections import defaultdict
from torch.optim import Optimizer
import torchtext
import random
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence
import logging
import gensim
import pickle

ROOT_PATH='../../data/'
ratings=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/user_action.csv')
# test_a=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/test_a.csv')
# test_a['date_']=15 # 由于 复赛数据B榜不可见 这里就不放入a榜数据使用
feed_info=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/feed_info.csv')
user_info=ratings.drop_duplicates('userid','first')[['userid','device']]
# 
print(ratings.shape)
ratings=ratings.drop_duplicates(['userid','feedid'],'last')
print(ratings.shape)

ACTION_LIST = ["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']#
PREDICT_LIST=["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']
def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df
feed2nid=pickle.load(open(ROOT_PATH+'tmp/feedid2nid.pkl','rb'))
feed_info=pd.read_pickle(ROOT_PATH+'tmp/cat_feed_info.pkl')

max_day = 15
df = ratings#pd.concat([ratings, test_a], axis=0, ignore_index=True)
df = df.merge(feed_info[['feedid', 'authorid', 'videoplayseconds','manual_keyword_id1','manual_tag_id1']], on='feedid', how='left')
## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000

df['is_finish'] = (df['play'] >= df['videoplayseconds']*0.92).astype('int8')
df.loc[df['play']>240000,'play']=240000
# df['play_times'] = (df['play'] / df['videoplayseconds']).astype('float16')
play_cols = ['is_finish']
df=reduce_mem(df)
del ratings
gc.collect()
n_day =12
for stat_cols in tqdm([  ['userid'],['feedid'],['authorid'], ['userid', 'authorid'],['userid', 'manual_tag_id1'],
        ['userid','manual_keyword_id1']]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1
        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
        if len(stat_cols)>1:
            fenmu=tmp.groupby('userid')['date_'].transform('count')
            tmp['{}_{}day_count'.format(f, n_day)]/=fenmu
        
        g = tmp.groupby(stat_cols)
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
#         if stat_cols[0]=='userid':
#             tmp['his_list']=tmp.groupby('userid')['feedid'].transform(history_arr)
#             feats.append('his_list')
        # 这部分类似目标编码了
        for y in PREDICT_LIST:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])
        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        tmp=reduce_mem(tmp)
        tmp.to_pickle(ROOT_PATH+'tmp/{}_feat_{}.pkl'.format(target_day,'_'.join(stat_cols)))
        del g, tmp
    gc.collect()
    
max_day=14
for stat_cols in tqdm([ ['userid'],['feedid'],['authorid'], ['userid', 'authorid'],['userid', 'manual_tag_id1'],
        ['userid','manual_keyword_id1']]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
#         tmp.to_pickle('./tmp/{}_feat_{}.pkl'.format(target_day,'_'.join(stat_cols)))
        tmp=pd.read_pickle(ROOT_PATH+'tmp/{}_feat_{}.pkl'.format(target_day,'_'.join(stat_cols)))
#         tmp=reduce_mem(tmp)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del tmp
    tmp_feat=stat_df.columns[len(stat_cols):]
    tmp_feat=tmp_feat.drop('date_')
    mean_tmp=stat_df[tmp_feat].mean(0)
    mean_tmp.fillna(-1,inplace=True)
    mean_tmp=mean_tmp.to_dict()
    pickle.dump(mean_tmp,open(ROOT_PATH+'tmp/{}_feat_mean.pkl'.format('_'.join(stat_cols)),'wb'))# 保存填充nan的均值
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    for kk,vv in mean_tmp.items():
        df[kk]=df[kk].fillna(vv) # 填充均值
    df=reduce_mem(df)
    del stat_df
    gc.collect()
df.fillna(-1,inplace=True)
feat=df.columns[18:]
print(len(feat))
df['reg']=np.sqrt((df['play']/df['videoplayseconds']).values)
pickle.dump(feat,open(ROOT_PATH+'tmp/feat_list.pkl','wb')) # 保存特征列表
normolizer_dict={}
for f in tqdm(feat):
    tmp=df[f].values.astype('float16').clip(-1,1e8)
    tmp_max=tmp.max() # 这里 或许我得保留均值和方差
    tmp_min=tmp.min()
    normolizer_dict[f+'_max']=tmp_max
    normolizer_dict[f+'_min']=tmp_min
    df[f]=((tmp-tmp_min)/tmp_max).astype('float16')   
df.to_pickle(ROOT_PATH+'tmp/ratings_feat_df.pkl')
pickle.dump(normolizer_dict,open(ROOT_PATH+'tmp/normolizer_dict.pkl','wb'))
