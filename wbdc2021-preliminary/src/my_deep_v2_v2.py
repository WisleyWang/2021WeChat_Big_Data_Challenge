#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# import os
# os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[3]:

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))

DATA_PATH=os.path.join(BASE_DIR, '../data/wedata/wechat_algo_data1/')
TMP_PATH=os.path.join(BASE_DIR, '../data/tmp/')
SUBMMIT_PATH=os.path.join(BASE_DIR, '../data/submission/')
MODEL_PATH=os.path.join(BASE_DIR, '../data/model/')

ratings=pd.read_csv(os.path.join(DATA_PATH,'user_action.csv'))
test_a=pd.read_csv(os.path.join(DATA_PATH,'test_a.csv'))
test_b=pd.read_csv(os.path.join(DATA_PATH,'test_b.csv'))
test_a['date_']=15
test_b['date_']=15
feed_info=pd.read_csv(os.path.join(DATA_PATH,'feed_info.csv'))
feed_embedding=pd.read_csv(os.path.join(DATA_PATH,'feed_embeddings.csv'))

graph_emb=np.load(os.path.join(TMP_PATH,'grap_embedding64_feedid_b1.npy'))
# In[4]:


ACTION_LIST = ["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']#
PREDICT_LIST=["read_comment","like", "click_avatar", "forward"]


# In[5]:


user_info=ratings.drop_duplicates('userid','first')[['userid','device']]


# In[6]:


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


# In[ ]:


# 这里就是要做整体的特征了
## 这里统一先用deedid 然后将feedid作为key来做映射
# ratings=pd.concat([ratings,test_a],axis=0)


# In[ ]:


# ratings.groupby('userid')['feedid'].ro


# In[7]:

feed_embedding['node']=feed_embedding['feedid'].apply(lambda x:feedid2nid[x])
feed_matrix=np.vstack(feed_embedding['feed_embedding'].apply(lambda x:np.array(list(map(float,x.strip().split(' '))))).values)

feed_emb=np.zeros(feed_matrix.shape)
indexs=feed_embedding['node'].values
for i in range(feed_matrix.shape[0]):
    feed_emb[indexs[i]]=feed_matrix[i]
del feed_embedding,feed_matrix
gc.collect()


# In[ ]:


# np.save('./tmp/knn_feed.npy',y_pred)


# In[8]:


def machine_tag(ma):
    s=sorted(list(map(lambda x :x.split(' '),ma.split(';'))),key=lambda x:float(x[1]))
    s=[i[0] for i in s[-2:]]
    return '_'.join(s)

feed_info['machine_tag_list']=feed_info['machine_tag_list'].fillna('0 0')
tmp=feed_info['machine_tag_list'].astype('str').apply(machine_tag)
feed_info['machine_tag_id1']=tmp.apply(lambda x:x.split('_')[0])
#feed_info['machine_tag_id2']=tmp.apply(lambda x:x.split('_')[1] if len(x.split('_'))>1 else x.split('_')[0])


# In[9]:


def manual_keyword(ma):
    s=ma.split(';')
    return '_'.join(s)

feed_info['manual_keyword_list']=feed_info['manual_keyword_list'].fillna('0;0')

tmp=feed_info['manual_keyword_list'].astype('str').apply( manual_keyword)
feed_info['manual_keyword_id1']=tmp.apply(lambda x:x.split('_')[0])
#feed_info['manual_keyword_id2']=tmp.apply(lambda x:x.split('_')[1] if len(x.split('_'))>1 else x.split('_')[0])


# In[10]:


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

feed_info['knn_feed']=np.load(os.path.join(TMP_PATH,'feed_4096_knn.npy'))


from sklearn.feature_extraction import FeatureHasher
hashing = FeatureHasher(n_features=16,input_type='string')
dense_arry1=hashing.transform(feed_info['manual_tag_list'].astype('str').values).toarray()
hashing = FeatureHasher(n_features=16,input_type='string')
dense_arry2=hashing.transform(feed_info['machine_tag_list'].astype('str').values).toarray()
hashing = FeatureHasher(n_features=8,input_type='string')
dense_arry3=hashing.transform(feed_info['machine_keyword_list'].astype('str').values).toarray()
hashing = FeatureHasher(n_features=8,input_type='string')
dense_arry4=hashing.transform(feed_info['manual_keyword_list'].astype('str').values).toarray()

max_day = 15
df = pd.concat([ratings, test_a,test_b], axis=0, ignore_index=True)

df = df.merge(feed_info[['feedid', 'authorid', 'videoplayseconds','bgm_song_id','manual_tag_id1','manual_keyword_id1','knn_feed']], on='feedid', how='left')
## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000
## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）

df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')

df['play_times'] = df['play'] / df['videoplayseconds']

play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]


## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day =11
for stat_cols in tqdm([  ['userid'],['feedid'],['authorid'], ['userid', 'authorid'],['userid', 'bgm_song_id'],
        ['userid','manual_keyword_id1'],['userid','manual_tag_id1']]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1

        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)

        tmp['date_'] = target_day

        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

        g = tmp.groupby(stat_cols)

        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')

        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        # 这里是对播放进行的统计 但是我觉得无用因为test无法体现
#         for x in play_cols[1:]:
#             for stat in ['max', 'mean']:
#                 tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)

#                 feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        # 这部分类似目标编码了
        for y in PREDICT_LIST:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')

            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])
        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)

        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    gc.collect()

df.fillna(-1,inplace=True)
feat=df.columns[21:]
print(len(feat))
mms = MinMaxScaler(feature_range=(0, 1))
df[feat] = mms.fit_transform(df[feat])

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
feed_data['hash_dense']=torch.from_numpy(np.hstack([dense_arry1,dense_arry2,dense_arry4,dense_arry3]).astype('float32'))
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

# In[19]:

# 文本特征处理
item_texts={}
user_texts={}
for f in ['manual_tag_list','manual_keyword_list','machine_keyword_list','asr','description','ocr']:#ocr
    feed_info[f]=feed_info[f].astype('str').apply(lambda x:x.replace(';',' '))
    item_texts[f]=feed_info[f].values

#  构建词典
import torchtext
class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos), input_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        self.emb.weight[:] = field.vocab.vectors
        self.proj = nn.Linear(input_dims, hidden_dims)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

        disable_grad(self.emb) # 词向量不可训练

    def forward(self, x):
        """
        x: (batch_size, max_length) LongTensor
        length: (batch_size,) LongTensor
        """
        x = self.emb(x).sum(1)# / length.unsqueeze(1).float() # 归一化
        return self.proj(x)

class BagOfWords(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()
        self.att_emb=Attn(hidden_dims) # 补充att
        self.emb = nn.Embedding(
            len(field.vocab.itos), hidden_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x):
        return self.att_emb(self.emb(x))#.mean(1)#/ length.unsqueeze(1).float() # 归一化
# 自建的att
class text_emb(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.att_emb=Attn(weight.shape[1])
        self.emb = nn.Embedding(
            weight.shape[0],weight.shape[1],
            padding_idx=0)
#         nn.init.xavier_uniform_(self.emb.weight)
        self.emb.weight.data.copy_(torch.from_numpy(weight).float())
#         self.emb.requires_grad_=False
    def forward(self, x):
        return self.att_emb(self.emb(x))#.mean(1)#/ length.unsqueeze(1).float() # 归一化

tokenize = lambda x: x.split(' ')
fields = {}
examples = []
for key, texts in item_texts.items():
    if  key in ['ocr','asr','description']:
        fields[key] = torchtext.data.Field(include_lengths=True, lower=True,tokenize=tokenize, batch_first=True, fix_length=64)
    else:
        fields[key] = torchtext.data.Field(include_lengths=True, lower=True,tokenize=tokenize, batch_first=True, fix_length=5)
    
for i in range(len(feed_info)):
    example = torchtext.data.Example.fromlist(
        [item_texts[key][i] for key in item_texts.keys()],
        [(key, fields[key]) for key in item_texts.keys()])  #( [feat1,feat2], [(key1,field1),(key2,field2)] )
    examples.append(example)
textset = torchtext.data.Dataset(examples, fields)
for key, field in fields.items():
    field.build_vocab(getattr(textset, key))


# In[21]:


for field_name, field in textset.fields.items():
    examples = [getattr(textset[i], field_name) for i in range(len(feed_info))]

    tokens, lengths = field.process(examples)

    if not field.batch_first:
        tokens = tokens.t()
    # 给feed +上文本向量
    feed_data[field_name] = tokens


class Model(nn.Module):
    def __init__(self,usr_data,feed_data,feed_embed,graph_emb):
        super().__init__()
        self.feed_data=feed_data
        self.user_data=user_data
        user_dict={'device':2,'userid':128}
        feed_dict={'bgm_song_id':16, 'bgm_singer_id':16,'authorid':16,'dense':32,'hash_dense':32
       ,'manual_keyword_id1':16,'manual_tag_id1':16,'machine_keyword_id1':16
            ,'machine_tag_id1':16,'knn_feed':16,
           'manual_tag_list':32,'manual_keyword_list':32,'machine_keyword_list':32,'asr':32,'description':32,'ocr':32
                  }
        self.model_dict=_init_input_modules(user_data,feed_data, user_dict,feed_dict)
        self.spare_liner=nn.Linear(8*16,128)
        self.dense_liner=nn.Linear(32*2,128)
        self.text_liner=nn.Linear(32*6+512+64,128)
        self.feed_embed= nn.Parameter(torch.from_numpy(feed_embed).float(),requires_grad=False)
        self.graph= nn.Parameter(torch.from_numpy(graph_emb).float(),requires_grad=False)
        self.reg_liner=nn.Linear(sum(user_dict.values())+128*3+64,1)
        self.dynami_dense=nn.Linear(70,64)
        self.cross1=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross2=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross3=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross4=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
#         self.dnn=DNN(sum(user_dict.values())+128*3+64,(128,128),dropout_rate=0.1)
        self.mmoe=MMOELayer(sum(user_dict.values())+128*3+64, mmoe_hidden_dim=128,num_task=4,n_expert=4,expert_activation=None)
        
        self.liner1=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
        self.liner2=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
        self.liner3=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
        self.liner4=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
    def forward(self,userid,feedid,batch_dense,is_train=True):
        user_projections=[]
#         feed_projections=[]
        dense_embedding=[]
        sparse_embedding=[]
        text_embedding=[]
        for feature, data in self.user_data.items():
            module = self.model_dict[feature]
            result = module(data)
            user_projections.append(result)
        for feature, data in self.feed_data.items():
#             print(feature)
            module = self.model_dict[feature]
            result = module(data)
            if result.shape[-1]==16:
                sparse_embedding.append(result)
            elif 'dense' in feature:
                dense_embedding.append(result)
            else:
#                 print(result.shape)
                text_embedding.append(result)
#         print(user_projections)

        user_feat=torch.cat(user_projections,-1)
        spare_emb=self.spare_liner(torch.cat(sparse_embedding,-1))
        dense_emb=self.dense_liner(torch.cat(dense_embedding,-1))
        text_emb=self.text_liner(torch.cat(text_embedding+[self.feed_embed,self.graph],-1))  
        feed_feat=torch.cat([spare_emb,dense_emb,text_emb],-1) #128*3
        dynami_dense=self.dynami_dense(batch_dense)
        combine=torch.cat([user_feat[userid],feed_feat[feedid],dynami_dense],axis=-1)
        cross1=self.cross1(combine)
        cross2=self.cross2(combine)
        cross3=self.cross3(combine)
        cross4=self.cross4(combine)
        outs=self.mmoe(combine)

        
        logit_gnn1=self.liner1(torch.cat([outs[0],cross1],axis=-1))#+ffm1#128+1+128*2
        logit_gnn2=self.liner2(torch.cat([outs[1],cross2],axis=-1))#+ffm2
        
        logit_gnn3=self.liner3(torch.cat([outs[2],cross3],axis=-1))#+ffm3
        logit_gnn4=self.liner4(torch.cat([outs[3],cross4],axis=-1))#+ffm4
        return logit_gnn1,logit_gnn2,logit_gnn3,logit_gnn4
    
def _init_input_modules(user_data,feed_data, user_dict,feed_dict):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()
    for column, data in user_data.items():
        if column in user_texts.keys():
            continue
        if data.dtype == torch.float32: # 数值类型的特征
            assert data.ndim == 2
            m = nn.Linear(data.shape[1],user_dict[column]) # 数值特征 做个线性变换
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif data.dtype == torch.int64:
            assert data.ndim == 1  # 整形的单值特征做个embedding
            m = nn.Embedding(data.max() + 2, user_dict[column], padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
        module_dict[column] = m  # 不同的特征名字对应不同的处理moderl 这里或许可以加FM进去
    
    for column, data in feed_data.items():
        if column in item_texts.keys():
            continue
        if column =='manuual_tag_list_emb':
            continue
        if data.dtype == torch.float32: # 数值类型的特征
            assert data.ndim == 2
            m = nn.Linear(data.shape[1],feed_dict[column]) # 数值特征 做个线性变换
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif data.dtype == torch.int64:
            assert data.ndim == 1  # 整形的单值特征做个embedding
            m = nn.Embedding(data.max() + 2, feed_dict[column], padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
        module_dict[column] = m  # 不同的特征名字对应不同的处理moderl 这里或许可以加FM进去
        
    if textset is not None:
        for column, field in textset.fields.items():
            if field.vocab.vectors:
                module_dict[column] = BagOfWordsPretrained(field,feed_dict[column])
            else:
                module_dict[column] = BagOfWords(field,feed_dict[column])
#     if userset is not None:
#         for column, field in userset.fields.items():
#             if field.vocab.vectors:
#                 module_dict[column] = BagOfWordsPretrained(field,user_dict[column])
#             else:
#                 module_dict[column] = BagOfWords(field,user_dict[column])
#     module_dict['manual_tag_user_read_comment_list']=module_dict['manual_tag_list']
    return module_dict

class MMOELayer(nn.Module):
    def __init__(self, hidden_size, mmoe_hidden_dim=128,num_task=4,n_expert=3,expert_activation=None):
        super(MMOELayer, self).__init__()
         # experts
        self.num_task=num_task
        self.expert_activation = expert_activation
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert).cuda(), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert).cuda(), requires_grad=True)
        # gates
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True).cuda() for _ in range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True).cuda() for _ in range(num_task)]
        for i in range(num_task):
            setattr(self, 'task_{}_dnn'.format(i+1),DNN(mmoe_hidden_dim,(128,128),dropout_rate=0.3,use_bn=True))
    def forward(self,x):
         # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', x, self.experts) # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        
        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum('ab, bc -> ac',x, gate) # batch * num_experts
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1) # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(experts_out) # batch * mmoe_hidden_size * num_experts
            outs.append(torch.sum(weighted_expert_output, 2)) # batch * mmoe_hidden_size
          # task tower
        task_outputs = list()
        for i in range(self.num_task):
            oo = outs[i]
            mod=getattr(self, 'task_{}_dnn'.format(i+1))
            oo = mod(oo)
            task_outputs.append(oo)
        
        return task_outputs

class HighwayMLP(nn.Module):
    def __init__(self,
                 input_size,
                 gate_bias=-3,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = DNN(input_size,(input_size,input_size,input_size),dropout_rate=0.1)

        self.gate_layer = nn.Linear(input_size,input_size)

        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)
class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

from torch.optim.lr_scheduler import LambdaLR
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
        Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
        Linearly decreases the multiplicative variable from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
class Attn(nn.Module):
    def __init__(self,hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size,1)
    def forward(self, x):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''   
        att=self.attn(x)
        att=F.tanh(att)
        att=F.softmax(att,1)
        att_x=att*x
        return att_x.sum(1)   
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)
        return loss
    
def build_optimizer(model, train_steps, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False, eps=1e-8)
    optimizer = Lookahead(optimizer, 5, 1)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * 0.1, t_total=train_steps)
    return optimizer, scheduler

def n_evaluate_nn(val_df,action_list,batch_size=512):
    model.eval()
    leng=len(val_df)
    val_src=val_df['userid'].apply(lambda x:userid2nid[x]).tolist()
    val_dst=val_df['feedid'].apply(lambda x:feedid2nid[x]).tolist()
    val_dense=torch.from_numpy(val_df[feat].values).float()
#     regs=torch.from_numpy(train_ratings['reg'].values).float()
    val_pred=[]
    all_aucs=[]
    weights=[0.4,0.3,0.2,0.1]
    with torch.no_grad():
        for i in tqdm(range(0,leng//batch_size+1)):
            #         print(i*batch_size,(i+1)*batch_size)
            batch_src=val_src[i*batch_size:(i+1)*batch_size]
            batch_dst=val_dst[i*batch_size:(i+1)*batch_size]
            batch_dense=val_dense[i*batch_size:(i+1)*batch_size].to(torch.device('cuda'))
            pred=model(batch_src,batch_dst,batch_dense)
            val_pred.append(torch.cat(pred,axis=-1).sigmoid().cpu().numpy())
        val_pred=np.concatenate(val_pred,axis=0)
        for i,action in enumerate(action_list):
            val_df['pred_'+action]=val_pred[:,i]
            label_nunique = val_df.groupby(by='userid')[action].transform('nunique')
            tmp_df = val_df[label_nunique == 2]
            aucs = tmp_df.groupby(by='userid').apply(
                lambda x: roc_auc_score(x[action].values, x['pred_'+action].values))
            all_aucs.append(np.mean(aucs))
            print('val %s uauc:'%action,np.mean(aucs))
            print('val %s auc:'%action,roc_auc_score(val_df[action].values,val_pred[:,i]))
        print('score uauc:',sum([all_aucs[i]*weights[i] for i in range(len(action_list))]))
        
def evaluate_nn(val_df,action,batch_size=512):
    model.eval()
    leng=len(val_df)
    val_src=val_df['userid'].apply(lambda x:userid2nid[x]).tolist()
    val_dst=val_df['feedid'].apply(lambda x:feedid2nid[x]).tolist()
    val_pred=[]
    with torch.no_grad():
        for i in tqdm(range(0,leng//batch_size+1)):
            #         print(i*batch_size,(i+1)*batch_size)
            batch_src=val_src[i*batch_size:(i+1)*batch_size]
            batch_dst=val_dst[i*batch_size:(i+1)*batch_size]

            pred=model(batch_src,batch_dst)

            val_pred.append(pred.sigmoid().view(-1).cpu().numpy())
        val_pred=np.concatenate(val_pred,axis=-1)
        val_df['pred_'+action]=val_pred
        label_nunique = val_df.groupby(by='userid')[action].transform('nunique')
        tmp_df = val_df[label_nunique == 2]
        
        aucs = tmp_df.groupby(by='userid').apply(
            lambda x: roc_auc_score(x[action].values, x['pred_'+action].values))
        print('val uauc:',np.mean(aucs))
        print('val auc:',roc_auc_score(val_df[action].values,val_pred))


# In[23]:


for f,d in user_data.items():
    user_data[f]=d.to(torch.device('cuda'))
    
for f,d in feed_data.items():
    feed_data[f]=d.to(torch.device('cuda'))


# In[24]:


def test_pred_func(model): 
    test_src=test_a['userid'].apply(lambda x:userid2nid[x]).tolist()
    test_dst=test_a['feedid'].apply(lambda x:feedid2nid[x]).tolist()
    batch_size=868
    test_dense=torch.from_numpy(test_a[feat].values).float()
    test_pred=[]
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0,len(test_a)//batch_size+1)):
    #         print(i*batch_size,(i+1)*batch_size)
            batch_src=test_src[i*batch_size:(i+1)*batch_size]
            batch_dst=test_dst[i*batch_size:(i+1)*batch_size]
            batch_dense=test_dense[i*batch_size:(i+1)*batch_size].cuda()
            pred=model(batch_src,batch_dst,batch_dense)
            pred=torch.cat(pred,axis=-1)
            test_pred.append(pred.sigmoid().cpu().numpy())
    test_pred=np.concatenate(test_pred,axis=0)
    return test_pred



import warnings
warnings.filterwarnings("ignore")


torch.cuda.empty_cache()
gc.collect()

max_day=15

train_ratings=df[(df.date_<max_day)]
test_a=df[df.date_==15]
# val_ratings=df[df.date_==max_day] #线下验证
# (~val_ratings.feedid.isin(train_ratings.feedid)).sum()/val_ratings.shape[0]


graph_emb=np.load('../data/tmp/grap_embedding64_feedid_b1.npy')

batch_size=4096
epochs=2
trn_dense=torch.from_numpy(train_ratings[feat].values).float()
model = Model(user_data,feed_data
             ,feed_embed=feed_emb,graph_emb=graph_emb) #in_features, hidden_features, out_features, rel_names
model=model.to(torch.device('cuda'))

train_steps = int(len(train_ratings) * epochs / batch_size) + 1
optimizer, scheduler = build_optimizer(model, train_steps, learning_rate=1e-3)
all_pred=[]
src=train_ratings['userid'].apply(lambda x: userid2nid[x]).values
dst=train_ratings['feedid'].apply(lambda x: feedid2nid[x]).values
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
        batch_dense=trn_dense[batch].to(torch.device('cuda'))
        batch_src=src[batch]
        batch_dst=dst[batch]
        
#         print(batch_src)
        logits = model(batch_src,batch_dst,batch_dense)
        batch_label=labels[batch].cuda()
        loss=criti(logits[0][:,0],batch_label[:,0])*0.8+criti(logits[1][:,0],batch_label[:,1])*0.8+        criti(logits[2][:,0],batch_label[:,2])*0.4+criti(logits[3][:,0],batch_label[:,3])*0.4
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
                print('train %s auc:'%aa,roc_auc_score(batch_label[:,ii],pred[:,ii]))
#             print(pred[:10])
        #if (epoch>0) and (ind in [500,1000,1608]):
    print('epoch %d  loss: %f '%(epoch,epoch_loss/(len(batch_index)//batch_size+1)))
#     n_evaluate_nn(val_df=val_ratings,action_list=PREDICT_LIST,batch_size=2048)
    
torch.save(model,model,os.path.join(MODEL_PATH,'my_deep_v2_v2.pth')

# inference
test_pred=test_pred_func(model)
test_a[PREDICT_LIST]=test_pred
# 取出test_b部分
sub=test_a[-len(test_b):][['userid','feedid']+PREDICT_LIST]
# 保存
sub.to_csv(os.path.join(SUBMMIT_PATH,'my_deep_v2_v2_b1.csv'),index=False)



