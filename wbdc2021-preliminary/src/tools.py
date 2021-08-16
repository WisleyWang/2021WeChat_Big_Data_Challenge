import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from configLightgbm import *
import gc


def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
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


def readData(readTestA=False):
    """
    读取数据
    """
    ## 读取训练集
    userDF = pd.read_csv(USER_ACTION_PATH)

    ## 读取测试集
    testDF = pd.read_csv(TEST_FILE_PATH)
    testDF['date_'] = MAX_DAY
    if readTestA:
        testDFA = pd.read_csv(TESTA_FILE_PATH)
        testDFA['date_'] = MAX_DAY



    ## 读取视频信息表
    feedDF = pd.read_csv(FEED_INFO_PATH)
    feedDF = feedDF[['feedid', 'authorid', 'videoplayseconds',"bgm_song_id","bgm_singer_id",
                    'manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list',
                    'description','ocr','asr']]

    ## 合并处理
    if readTestA:
        testDFA = pd.read_csv(TESTA_FILE_PATH)
        testDFA['date_'] = MAX_DAY
        dataDF = pd.concat([userDF, testDF], axis=0, ignore_index=True)
    else:
        dataDF = pd.concat([userDF, testDF], axis=0, ignore_index=True)
        dataDF = pd.concat([userDF, testDF,testDFA], axis=0, ignore_index=True)
    uselen=userDF.shape[0]+testDF.shape[0]

    dataDF = dataDF.merge(feedDF, on='feedid', how='left')
    ## 读取embed数据
    embedDF = pd.read_csv(FEED_EMBEDDINGS_PATH)

    del userDF,testDF,feedDF
    gc.collect()

    ## 视频时长是秒，转换成毫秒，才能与play、stay做运算
    dataDF['videoplayseconds'] *= 1000

    ## bgm_song_id、bgm_singer_id中的NAN和float处理
    dataDF['bgm_song_id']=dataDF['bgm_song_id'].fillna(-1)
    dataDF['bgm_singer_id']=dataDF['bgm_singer_id'].fillna(-1)
    dataDF['bgm_song_id']=dataDF['bgm_song_id'].astype('int')
    dataDF['bgm_singer_id']=dataDF['bgm_singer_id'].astype('int')

    return dataDF,uselen



def str2int(string):
    try:
        string_int = int(string)
        return string_int
    except ValueError:
        return -1

def countKeyTag(x):
    if x is np.nan:
        return 0
    else:
        return len(x.split(';'))

def firstKeyOrTag(x):
    if x is np.nan:
        return -1
    else:
        return str2int(x.split(';')[0])

def calFreqKeyOrTag(df):
    """计算频率"""
    freq=dict()
    for ks in df:
        if not ks is np.nan:
            for k in [str2int(i) for i in ks.split(';')]:
                freq[k]=freq.get(k,0)+1
    return freq


def calFreqKeyOrTagOfAuthor(df):
    """计算频率"""
    freq=dict()
    for ks in df:
        if not ks is np.nan:
            for k in [str2int(i) for i in ks.split(';')]:
                freq[k]=freq.get(k,0)+1
    return freq


def maxFreqKeyOrTag(x,freq):
    """
    返回频率最高的key or tag
    """
    if x is np.nan:
        return -1
    else:
        ks=[str2int(i) for i in x.split(';')]
        ksFreq=[freq.get(i,0) for i in ks]
        return ks[ksFreq.index(max(ksFreq))]

def minFreqKeyOrTag(x,freq):
    """
    返回频率最高的key or tag
    """
    if x is np.nan:
        return -1
    else:
        ks=[str2int(i) for i in x.split(';')]
        ksFreq=[freq.get(i,0) for i in ks]
        return ks[ksFreq.index(min(ksFreq))]

def maxFreqKeyOrTagOfGroup(x):
    freq=dict()
    for ks in x :
        if not ks is np.nan:
            for k in [str2int(i) for i in ks.split(';')]:
                freq[k]=freq.get(k,0)+1
        else:
            freq[-1]=freq.get(-1,0)+1
    
    return max(freq, key=freq.get)

def find_tag(x):
    if x is np.nan:
        return -1
    max_prob = 0
    for i in x.split(';'):
        if len(i.split())<2:
            print(i,x)
        key = int(i.split()[0])
        prob = float(i.split()[1])
        if prob >= max_prob:
            max_prob = prob
            max_key = key
    return max_key


def getEmbed(x,f):
    if len(x[x[f]==1])!=0 and len(x[x[f]==0])!=0:
        return (x[x[f]==1].mean()-x[x[f]==0].mean())/2
    elif len(x[x[f]==1])==0:
        return -x[x[f]==0].mean()
    else:
        return x[x[f]==1].mean()

def getTextMachine(x):
    if x is np.nan:
        return []
    return [i.split(' ')[0] for i in x.split(';')]

def getText(x):
    if x is np.nan:
        return []
    return [i for i in x.split(';')]


def find_tag(x):
    if x is np.nan:
        return -1
    max_prob = 0
    for i in x.split(';'):
        key = int(i.split()[0])
        prob = float(i.split()[1])
        if prob >= max_prob:
            max_prob = prob
            max_key = key
    return max_key

def getvec(model,x):
    if x in model:
        return model.wv[x]
    return np.zeros(32)

def getEmbed(x,actions,model):
    actionIdx=(x[actions[0]]==1)|(x[actions[1]]==1)|(x[actions[2]]==1)|(x[actions[3]]==1)
    noactionIdx=~actionIdx
    vec=np.zeros(32)
    if sum(actionIdx)>0:
        vecAc=x[actionIdx]['tag_maxprob'].apply(lambda x:getvec(model,str(x))).mean()
        vec=vec+np.array(vecAc)
    if sum(noactionIdx)>0:
        vecAcno=x[noactionIdx]['tag_maxprob'].apply(lambda x:getvec(model,str(x))).mean()
        vec=vec-np.array(vecAcno)
    return pd.Series(vec,index=['tag_maxprob_w2v_'+str(i) for i in range(32)])

def cosSim(v1,v2):
    return 1-np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def calSim(user,date,feed,userVec,model):
    if date==1 or (not user in userVec[date]) or (not str(feed) in model.wv) :
        return 1
    return cosSim(userVec[date][user],model.wv[str(feed)])

def calSim1(user,date,feed,userVec,feedidGraphEmbed):
    if date==1 or (not user in userVec[date]) or (not str(feed) in feedidGraphEmbed) :
        return 1
    return cosSim(userVec[date][user],feedidGraphEmbed[str(feed)])

## 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)
    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag
    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc