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
test_a=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/test_a.csv')
test_a['date_']=15 # 由于 复赛数据B榜不可见 这里就不放入a榜数据使用
feed_info=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/feed_info.csv')
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
##----------------------保存suer2id,feed2id
pickle.dump(feedid2nid,open(ROOT_PATH+'tmp/feedid2nid.pkl','wb'))
pickle.dump(userid2nid,open(ROOT_PATH+'tmp/userid2nid.pkl','wb'))
##----------------注意!!
df=pd.concat([ratings[['userid','feedid']],test_a[['userid','feedid']]],axis=0)

df['feedid_node']=df['feedid'].apply(lambda x:feedid2nid[x])
tmp=df.groupby('userid')['feedid_node'].apply(lambda x: ' '.join([str(i) for i in x]))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if not os.path.exists(ROOT_PATH+'tmp/w2v_feedid_32_hs2.txt'):
    w2v=Word2Vec(tmp.apply(lambda x: x.split(' ')).tolist(),vector_size=32, window=10, epochs=10, min_count=2,
                         hs=1, sample=0.0001, workers=6 , seed=1018)
    #window12 6 16
    w2v.wv.save_word2vec_format(ROOT_PATH+'tmp/w2v_feedid_32_hs2.txt')
w2v=gensim.models.KeyedVectors.load_word2vec_format(
        ROOT_PATH+'/tmp/w2v_feedid_32_hs2.txt', binary=False)

feed_emb=np.zeros((len(feed_info),32))
for i in range(len(feed_info)):
    if str(i) not in w2v.key_to_index:
        continue
    feed_emb[i]=w2v[str(i)]
np.save(ROOT_PATH+'tmp/grap_embedding32_hs2.npy',feed_emb)  
print('---------------grap_embedding32_hs2.npy have save-------------')
#--------------------------------------
if not os.path.exists(ROOT_PATH+'tmp/w2v_feedid_32_sg2.txt'):
    w2v=Word2Vec(tmp.apply(lambda x: x.split(' ')).tolist(),vector_size=32, window=12, epochs=10, min_count=2,
                         sg=1, sample=0.0001, workers=6 , seed=1017)
    #window12 6 16
    w2v.wv.save_word2vec_format(ROOT_PATH+'tmp/w2v_feedid_32_sg2.txt')
w2v=gensim.models.KeyedVectors.load_word2vec_format(
        ROOT_PATH+'tmp/w2v_feedid_32_sg2.txt', binary=False)

feed_emb=np.zeros((len(feed_info),32))
for i in range(len(feed_info)):
    if str(i) not in w2v.key_to_index:
        continue
    feed_emb[i]=w2v[str(i)]

np.save(ROOT_PATH+'tmp/grap_embedding32_sg2.npy',feed_emb)  
del feed_emb,w2v,tmp
gc.collect()
print('---------------grap_embedding32_sg2.npy have save-------------')
#生成feed embedding 矩阵------------------------------------------------------------------------------------
try:
    feed_embedding=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/feed_embeddings.csv')
except:
    print('not find '+ ROOT_PATH+'wedata/wechat_algo_data2/feed_embeddings.csv')
if not os.path.exists(ROOT_PATH+'tmp/feed_emb.npy'):
    feed_embedding['node']=feed_embedding['feedid'].apply(lambda x:feedid2nid[x])
    feed_matrix=np.vstack(feed_embedding['feed_embedding'].apply(lambda x:np.array(list(map(float,x.strip().split(' '))))).values)

    feed_emb=np.zeros(feed_matrix.shape)
    indexs=feed_embedding['node'].values
    for i in range(feed_matrix.shape[0]):
        feed_emb[indexs[i]]=feed_matrix[i]
    # 保存
    np.save(ROOT_PATH+'tmp/feed_emb.npy',feed_emb)
    del feed_matrix,feed_emb
    gc.collect()
    print(' -------------- feed_emb.npy have save ----------------')
else:
    print('feed_emb.npy had save before!')
del feed_embedding
gc.collect()
### 生成交互特征---------------------------------------------------------------------------
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

def machine_tag(ma):
    s=sorted(list(map(lambda x :x.split(' '),ma.split(';'))),key=lambda x:float(x[1]))
    s=[i[0] for i in s[-2:]]
    return '_'.join(s)

feed_info['machine_tag_list']=feed_info['machine_tag_list'].fillna('0 0')
tmp=feed_info['machine_tag_list'].astype('str').apply(machine_tag)
feed_info['machine_tag_id1']=tmp.apply(lambda x:x.split('_')[0])
def manual_keyword(ma):
    s=ma.split(';')
    return '_'.join(s)

feed_info['manual_keyword_list']=feed_info['manual_keyword_list'].fillna('0;0')

tmp=feed_info['manual_keyword_list'].astype('str').apply( manual_keyword)
feed_info['manual_keyword_id1']=tmp.apply(lambda x:x.split('_')[0])
#feed_info['manual_keyword_id2']=tmp.apply(lambda x:x.split('_')[1] if len(x.split('_'))>1 else x.split('_')[0])
def manual_tag(ma):
    s=ma.split(';')
    return '_'.join(s)
feed_info['manual_tag_list']=feed_info['manual_keyword_list'].fillna('0;0')
tmp=feed_info['manual_tag_list'].astype('str').apply( manual_tag)
feed_info['manual_tag_id1']=tmp.apply(lambda x:x.split('_')[0])
#feed_info['manual_tag_id2']=tmp.apply(lambda x:x.split('_')[1] if len(x.split('_'))>1 else x.split('_')[0])


feed_info['machine_keyword_list']=feed_info['machine_keyword_list'].fillna('0;0')
tmp=feed_info['machine_keyword_list'].astype('str').apply( manual_tag)
feed_info['machine_keyword_id1']=tmp.apply(lambda x:x.split('_')[0])
#feed_info['machine_keyword_id2']=tmp.apply(lambda x:x.split('_')[1] if len(x.split('_'))>1 else x.split('_')[0])

feed_info['knn_feed']=np.load(ROOT_PATH+'tmp/knn_2550_feed.npy') # 之前本地聚类的结果
#-------------------------------------------------------------------------------
#为测试集构建拼接关系,这里直接预先保存
feed_info[['feedid', 'authorid', 'videoplayseconds','manual_keyword_id1','manual_tag_id1']].to_pickle(ROOT_PATH+'tmp/cat_feed_info.pkl')
#-------------------------------------------------------------------------------
from sklearn.feature_extraction import FeatureHasher
hashing = FeatureHasher(n_features=16,input_type='string')
dense_arry1=hashing.transform(feed_info['manual_tag_list'].astype('str').values).toarray()
hashing = FeatureHasher(n_features=16,input_type='string')
dense_arry2=hashing.transform(feed_info['machine_tag_list'].astype('str').values).toarray()
hashing = FeatureHasher(n_features=8,input_type='string')
dense_arry3=hashing.transform(feed_info['machine_keyword_list'].astype('str').values).toarray()
hashing = FeatureHasher(n_features=8,input_type='string')
dense_arry4=hashing.transform(feed_info['manual_keyword_list'].astype('str').values).toarray()
hashing = FeatureHasher(n_features=16,input_type='string')
dense_arry5=hashing.transform(feed_info['description_char'].astype('str').values).toarray()

###-----------保存交互特征----------------------
max_day = 15
df = ratings#pd.concat([ratings, test_a], axis=0, ignore_index=True)
df = df.merge(feed_info[['feedid', 'authorid', 'videoplayseconds','bgm_song_id','manual_keyword_id1',]], on='feedid', how='left')
## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000
df['is_finish'] = (df['play'] >= df['videoplayseconds']*0.92).astype('int8')
df.loc[df['play']>240000,'play']=240000
# df['play_times'] = (df['play'] / df['videoplayseconds']).astype('float16')
play_cols = ['is_finish']
df=reduce_mem(df)
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
#--------------------加载交互特征--------------------
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
print('-------开始构建 user 与feed 数据表------')
feed_data={}
user_data={}
### 给节点传入特征
feed_feats=['feedid','authorid','videoplayseconds','bgm_song_id','bgm_singer_id','description_char'
            ,'manual_keyword_id1','manual_tag_id1','machine_keyword_id1'
            ,'machine_tag_id1','knn_feed']
feeds=feed_info[feed_feats]

for f in feed_feats:
    feeds[f]=feeds[f].fillna(0)

# 对id 重新进行编码，类别特征
for f in ['bgm_song_id', 'bgm_singer_id','authorid','knn_feed']:
    gens=LabelEncoder()
    feeds[f]=gens.fit_transform(feeds[f].astype('str'))
    feed_data[f]=torch.from_numpy(feeds[f].values)

# 对于tag 和keyword 要联合编码
gens=LabelEncoder()
tmp=pd.concat([feeds[f].astype('str') for f in ['manual_tag_id1','machine_tag_id1'] ])
gens=gens.fit(tmp)
for f in ['manual_tag_id1','machine_tag_id1']:
    feeds[f]=gens.transform(feeds[f].astype('str'))
    feed_data[f]=torch.from_numpy(feeds[f].values)
    
gens=LabelEncoder()
tmp=pd.concat([feeds[f].astype('str') for f in ['manual_keyword_id1','machine_keyword_id1'] ])
gens=gens.fit(tmp)
for f in ['manual_keyword_id1','machine_keyword_id1']:
    feeds[f]=gens.transform(feeds[f].astype('str'))
    feed_data[f]=torch.from_numpy(feeds[f].values)
    
## 连续特征进行归一化
dense_features=['videoplayseconds']
for f in dense_features:
    feeds[f]=np.log(feeds[f] + 1.0)
mms = MinMaxScaler(feature_range=(0, 1))
feeds[dense_features] = mms.fit_transform(feeds[dense_features])
feed_data['dense']=torch.from_numpy(feeds[dense_features].values.astype('float32'))
# feed_data['manuual_tag_list_emb']=torch.from_numpy(seq).long()
feed_data['hash_dense']=torch.from_numpy(np.hstack([dense_arry1,dense_arry2,dense_arry4,dense_arry3,dense_arry5]).astype('float32'))
#------------------------------------------------------------------------------------------------
# user 
user_feats=['userid','device']
for f in user_feats:
    user_info[f]=user_info[f].fillna(0)
    
for f in ['device']:
    gens=LabelEncoder()
    user_info[f]=gens.fit_transform(user_info[f])
    user_data[f]=torch.from_numpy(user_info[f].values)   
# 传入userid
user_data['userid']=torch.from_numpy(user_info['userid'].apply(lambda x:userid2nid[x]).values)
print('---------开始构建文本特征----------------')
item_texts={}
for f in ['manual_tag_list','manual_keyword_list','machine_keyword_list','asr','description','ocr']:#ocr
    feed_info[f]=feed_info[f].astype('str').apply(lambda x:x.replace(';',' '))
    item_texts[f]=feed_info[f].values
    
#### 保存 feed_data,user_data,,examplesfields
pickle.dump(feed_data,open(ROOT_PATH+'tmp/feed_data.pkl','wb'))
print('save '+ROOT_PATH+'tmp/feed_data.pkl')
pickle.dump(user_data,open(ROOT_PATH+'tmp/user_data.pkl','wb'))
print('save '+ROOT_PATH+'tmp/user_data.pkl')
pickle.dump(item_texts,open(ROOT_PATH+'tmp/item_texts.pkl','wb'))
print('save '+ROOT_PATH+'tmp/item_texts.pkl')

del feed_data,user_data,item_texts
gc.collect()