import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm.sklearn import LGBMClassifier
from gensim.models import Word2Vec
from collections import defaultdict
import gc
import time
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

from configLightgbm import *
from tools import *


def getFeatureTagKeyword(dataDF,uselen):
    """
    生成tag和keyword特征
    """
    feats=[] #统计生成的特征
    for f in tqdm(['machine_tag_list','manual_tag_list','manual_keyword_list','machine_keyword_list']):
        freq=calFreqKeyOrTag(dataDF[f])
        dataDF[f+'_num']=dataDF[f].apply(lambda x:countKeyTag(x))
        dataDF[f+'_first']=dataDF[f].apply(lambda x:firstKeyOrTag(x))
        if f in ['machine_tag_list']:
            dataDF[f+'_maxprob'] = list(map(find_tag, dataDF[f].values))
            feats.extend([f+'_maxprob'])
        dataDF[f+'_max_freq']=dataDF[f].apply(lambda x:maxFreqKeyOrTag(x,freq))
        dataDF[f+'_min_freq']=dataDF[f].apply(lambda x:minFreqKeyOrTag(x,freq))
        
        g=dataDF.groupby('authorid')['manual_keyword_list'].apply(lambda x:maxFreqKeyOrTagOfGroup(x))
        dataDF[f+'_max_freq_of_authorid']=dataDF['authorid'].map(g)
        
        g=dataDF.groupby('userid')['manual_keyword_list'].apply(lambda x:maxFreqKeyOrTagOfGroup(x))
        dataDF[f+'_max_freq_of_userid']=dataDF['userid'].map(g)
        feats.extend([f+'_num',f+'_first',f+'_max_freq',f+'_min_freq',f+'_max_freq_of_authorid',f+'_max_freq_of_userid'])

    dataDF = reduce_mem(dataDF,list(dataDF.columns))
    dataDF.iloc[:uselen][feats].to_pickle(FEATURE_PATH+'tag_keyword_features_testab.pkl')
    dataDF[feats].to_pickle(FEATURE_PATH+'tag_keyword_features_testab_all.pkl')
    return dataDF


def getFeatureStatistic(dataDF,uselen):
    ##更加多少天之间的数据进行统计
    tmp=dataDF[dataDF['date_']<=15]
    feats=[]
    ## 各个id的count
    for f in tqdm(['userid', 'feedid', 'authorid','bgm_song_id','bgm_singer_id']):
        dataDF[f + '_count'] = dataDF[f].map(tmp[f].value_counts())
        feats.extend([f + '_count'])

    ## f1_in_f2_nunique 一个id有多少种另一个id对他曝光
    for f1, f2 in tqdm([
        ['userid', 'feedid'],
        ['userid', 'authorid'],
        ['userid', 'bgm_song_id'],
        ['userid', 'bgm_singer_id'],
        ['authorid', 'bgm_song_id'],##newadd
        ['authorid', 'bgm_singer_id'],##newadd
    ]):
        dataDF['{}_in_{}_nunique'.format(f1, f2)] =dataDF[f2].map(tmp.groupby(f2)[f1].agg('nunique'))
        dataDF['{}_in_{}_nunique'.format(f2, f1)] =dataDF[f1].map(tmp.groupby(f1)[f2].agg('nunique'))
        feats.extend(['{}_in_{}_nunique'.format(f1, f2),'{}_in_{}_nunique'.format(f2, f1)])
        
    dataDF['feedid_in_authorid_nunique'] = dataDF['authorid'].map(tmp.groupby('authorid')['feedid'].agg('nunique'))
    feats.extend(['feedid_in_authorid_nunique'])

    ## id1和id2共同出现的次数，id1和id2共同出现的次数/id1出现的次数 ，id1和id2共同出现的次数/id1出现的次数
    for f1, f2 in tqdm([
        ['userid', 'authorid'],
        ['userid', 'bgm_song_id'],
        ['userid', 'bgm_singer_id'],
        ['authorid', 'bgm_song_id'],##newadd
        ['authorid', 'bgm_singer_id'],##newadd
    ]):
        g=tmp.groupby([f1, f2])['date_'].agg('count')
        g=g.reset_index()
        g=g.rename(columns={'date_':'{}_{}_count'.format(f1, f2)})
        dataDF=dataDF.merge(g,on=[f1,f2], how='left')
        dataDF['{}_in_{}_count_prop'.format(f1, f2)] = dataDF['{}_{}_count'.format(f1, f2)] / (dataDF[f2 + '_count'] + 1)
        dataDF['{}_in_{}_count_prop'.format(f2, f1)] = dataDF['{}_{}_count'.format(f1, f2)] / (dataDF[f1 + '_count'] + 1)
        feats.extend(['{}_{}_count'.format(f1, f2),'{}_in_{}_count_prop'.format(f1, f2),'{}_in_{}_count_prop'.format(f2, f1)])
        
    ## 视频时长特征
    for t in tqdm(['mean','max','min','std','median']):  
        dataDF['videoplayseconds_in_userid_'+t] = dataDF['userid'].map(tmp.groupby('userid')['videoplayseconds'].agg(t)) #userid看的视频的视频时长特征
        dataDF['videoplayseconds_in_authorid_'+t] = dataDF['authorid'].map(tmp.groupby('authorid')['videoplayseconds'].agg(t)) #authorid的视频时长特征
        feats.extend(['videoplayseconds_in_userid_'+t,'videoplayseconds_in_authorid_'+t])
        
    ## 
    for t in tqdm(['mean','max','min','std']):
        dataDF['play_in_userid_'+t] = dataDF['userid'].map(tmp.groupby('userid')['play'].agg(t))
        dataDF['stay_in_userid_'+t] = dataDF['userid'].map(tmp.groupby('userid')['stay'].agg(t))
        dataDF['play_in_authorid_'+t] = dataDF['authorid'].map(tmp.groupby('authorid')['play'].agg(t))
        dataDF['stay_in_authorid_'+t] = dataDF['authorid'].map(tmp.groupby('authorid')['stay'].agg(t))
        feats.extend(['play_in_userid_'+t,'stay_in_userid_'+t,'play_in_authorid_'+t,'stay_in_authorid_'+t])

    dataDF = reduce_mem(dataDF,list(dataDF.columns))
    dataDF.iloc[:uselen][feats].to_pickle(FEATURE_PATH+'statisticFeature15_testab.pkl')
    dataDF[feats].to_pickle(FEATURE_PATH+'statisticFeature15_testab_all.pkl')
    return dataDF


def getFeatureWindows(dataDF,n_day):
    ## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
    dataDF['is_finish'] = (dataDF['play'] >= dataDF['videoplayseconds']*0.9).astype('int8')
    dataDF['play_times'] = dataDF['play'] / dataDF['videoplayseconds']
    dataDF['play_stay'] = dataDF['play'] / dataDF['stay']
    dataDF['stay_minus_play'] = dataDF['stay'] - dataDF['play']##newadd
    play_cols = ['is_finish', 'play_times','play_stay', 'play', 'stay','stay_minus_play']

    ## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
    features=[]
    for stat_cols in tqdm([['userid'],['feedid'],['authorid'],
                        ['manual_keyword_list_first'],['manual_tag_list_first'],
                        ['userid', 'authorid'],['userid','manual_keyword_list_first'],['userid','manual_tag_list_first'],
                        ['authorid','manual_keyword_list_first'],['authorid','manual_tag_list_first']]):
        f = '_'.join(stat_cols)
        stat_df = pd.DataFrame()
        for target_day in tqdm(range(2, MAX_DAY + 1)):
            left, right = max(target_day - n_day, 1), target_day - 1 #[left,right]是target_day前n天的天，[1,1]:1\\[1,2]:2\\...\\[1,5]:6\\...
            tmp = dataDF[((dataDF['date_'] >= left) & (dataDF['date_'] <= right))].reset_index(drop=True) # 取出target_day前n_day的数据
            tmp['date_'] = target_day
            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')# 每个id前n_day天的曝光
            g = tmp.groupby(stat_cols)
            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g['is_finish'].transform('mean') #每个id看完视频转化的比率
            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)] 
            #每个id的'play_times', 'play', 'stay'的'max', 'mean'
            for x in play_cols[1:]:
                for stat in ['min','max', 'mean','median','std']:
                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat) 
                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
            #每个id
            for y in (ACTION_LIST):
                tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
                tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y),])
            
            tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)#去重
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
            del g, tmp
        features.extend(feats) 
        dataDF = dataDF.merge(stat_df, on=stat_cols + ['date_'], how='left')
        del stat_df
        gc.collect()

    dataDF = reduce_mem(dataDF,list(dataDF.columns))
    dataDF[features].to_pickle(FEATURE_PATH+'windowsFeature{}day_testb.pkl'.format(n_day))


def getFeatureEmbedding(dataDF,uselen):
    embedDF = pd.read_csv(FEED_EMBEDDINGS_PATH)
    estimator = PCA(n_components=32)
    tmp=embedDF['feed_embedding'].apply(lambda x:[float(i) for i in x.split(' ')[:512]])
    arr=np.array(tmp.values.tolist())
    embedPCA32=estimator.fit_transform(arr)

    columnsEmbed=['embed_'+str(i) for i in range(len(embedPCA32[0]))]
    embed32DF=pd.DataFrame(embedPCA32,index=embedDF.index,columns=columnsEmbed)
    embed32DF['feedid']=embedDF['feedid']

    embed32DF.to_csv(FEATURE_PATH+'embedPCA32.csv',index=False)
    dataDF = dataDF.merge(embed32DF, on='feedid', how='left')
    dataDF = reduce_mem(dataDF,list(dataDF.columns))
    
    n_day = 5
    for stat_cols in tqdm([['userid']]):
        for action in tqdm(ACTION_LIST[:4]):
            f = '_'.join(stat_cols)
            stat_df = pd.DataFrame()
            for target_day in tqdm(range(2, MAX_DAY + 1)):
                left, right = max(target_day - n_day, 1), target_day - 1 #[left,right]是target_day前n天的天，[1,1]:1\\[1,2]:2\\...\\[1,5]:6\\...
                tmp = dataDF[((dataDF['date_'] >= left) & (dataDF['date_'] <= right))].reset_index(drop=True) # 取出target_day前n_day的数据
                tmp['date_'] = target_day

                columnsEmbed=['embed_'+str(i) for i in range(32)]
                columnsEmbed.append(action)
                g=tmp.groupby('userid')[columnsEmbed].apply(lambda x:getEmbed(x,action))
                g=g.drop(columns=action)
                columnsEmbed.remove(action)
                columnsUserEmbed=['userid_'+i+'_'+action for i in columnsEmbed]
                g.columns=columnsUserEmbed
                g.reset_index(level=0, inplace=True)
                g['date_']=tmp['date_']
                
                g = g[stat_cols + columnsUserEmbed + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)#去重
                stat_df = pd.concat([stat_df, g], axis=0, ignore_index=True)
                del g, tmp

            dataDF = dataDF.merge(stat_df, on=stat_cols + ['date_'], how='left')
            del stat_df
            gc.collect()

    dataDF = reduce_mem(dataDF,list(dataDF.columns))

    """
    计算相似度
    """
    for action in tqdm(ACTION_LIST[:4]):
        columnsEmbed=['embed_'+str(i) for i in range(32)]
        columnsUserEmbed=['userid_'+i+'_'+action for i in columnsEmbed]
        dataDF['cosSimEmbed_'+action]=(np.sum(dataDF[columnsEmbed].values*dataDF[columnsUserEmbed].values,axis=1))/(np.linalg.norm(dataDF[columnsEmbed].values,axis=1)*np.linalg.norm(dataDF[columnsUserEmbed].values,axis=1))

    columnsEmbed=['embed_'+str(i) for i in range(32)]
    columnsUserEmbed=['userid_'+i+'_'+action for i in columnsEmbed for action in ACTION_LIST[:4]]
    columnsSim=['cosSimEmbed_'+action for action in ACTION_LIST[:4]]
    columns=columnsEmbed+columnsUserEmbed+columnsSim

    dataDF[columns].to_pickle(FEATURE_PATH+'featureEmbedding_testab_all.pkl')
    dataDF.iloc[:uselen][columns].to_pickle(FEATURE_PATH+'featureEmbedding_testab.pkl')

def getFeatureTagKeywordW2v(dataDF):
    """
    训练Word2Vec向量，并保存模型
    """
    from gensim.models import Word2Vec
    texts_manual_keyword_list_first=dataDF.groupby(['userid','date_'])['manual_keyword_list_first'].apply(lambda x:list([str(i) for i in x]))
    model_manual_keyword_list_first = Word2Vec(texts_manual_keyword_list_first, sg=1, vector_size=32,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=32)

    texts_manual_tag_list_first=dataDF.groupby(['userid','date_'])['manual_tag_list_first'].apply(lambda x:list([str(i) for i in x]))
    model_manual_tag_list_first = Word2Vec(texts_manual_tag_list_first, sg=1, vector_size=32,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=32)

    # 保存模型
    model_manual_keyword_list_first.save(MODEL_PATH+"manual_keyword_list_first_word2vec_testb.model")
    model_manual_tag_list_first.save(MODEL_PATH+"manual_tag_list_first_word2vec_testb.model")

    """
    将w2v加入到dataDF中去
    """
    columnsW2vKeywordFirst=['w2v_keyword_first'+str(i) for i in range(32)]
    word_list = model_manual_keyword_list_first.wv.index_to_key
    w2vArr = np.array([model_manual_keyword_list_first.wv[word] for word in word_list])
    w2v32DF=pd.DataFrame(w2vArr,columns=columnsW2vKeywordFirst)
    w2v32DF['manual_keyword_list_first']=[int(i) for i in model_manual_keyword_list_first.wv.index_to_key]

    w2v32DF.to_csv(FEATURE_PATH+'w2v32_manual_keyword_first.csv',index=False)

    dataDF = dataDF.merge(w2v32DF, on='manual_keyword_list_first', how='left')
    dataDF = reduce_mem(dataDF,list(dataDF.columns))

    columnsW2vTagFirst=['w2v_tag_first'+str(i) for i in range(32)]
    word_list = model_manual_tag_list_first.wv.index_to_key
    w2vArr = np.array([model_manual_tag_list_first.wv[word] for word in word_list])
    w2v32DF=pd.DataFrame(w2vArr,columns=columnsW2vTagFirst)
    w2v32DF['manual_tag_list_first']=[int(i) for i in model_manual_tag_list_first.wv.index_to_key]

    w2v32DF.to_csv(FEATURE_PATH+'w2v32_manual_tag_list_first.csv',index=False)

    dataDF = dataDF.merge(w2v32DF, on='manual_tag_list_first', how='left')
    dataDF = reduce_mem(dataDF,list(dataDF.columns))

    """
    用w2v计算userid的w2v
    """

    n_day = 5
    for stat_cols in tqdm([['userid']]):
        for action in tqdm(ACTION_LIST[:4]):
            f = '_'.join(stat_cols)
            stat_df = pd.DataFrame()
            for target_day in tqdm(range(2, MAX_DAY + 1)):
                left, right = max(target_day - n_day, 1), target_day - 1 #[left,right]是target_day前n天的天，[1,1]:1\\[1,2]:2\\...\\[1,5]:6\\...
                tmp = dataDF[((dataDF['date_'] >= left) & (dataDF['date_'] <= right))].reset_index(drop=True) # 取出target_day前n_day的数据
                tmp['date_'] = target_day
                
                columnsW2vKeywordFirst=['w2v_keyword_first'+str(i) for i in range(32)]
                columnsW2vKeywordFirst.append(action)
                g=tmp.groupby('userid')[columnsW2vKeywordFirst].apply(lambda x:getEmbed(x,action))
                g=g.drop(columns=action)
                columnsW2vKeywordFirst.remove(action)
                columnsUserW2vKeywordFirst=['userid_'+i+'_'+action for i in columnsW2vKeywordFirst]
                g.columns=columnsUserW2vKeywordFirst
                g.reset_index(level=0, inplace=True)
                g['date_']=tmp['date_']
                g = g[stat_cols + columnsUserW2vKeywordFirst + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)#去重
                stat_df = pd.concat([stat_df, g], axis=0, ignore_index=True)
                
                columnsW2vTagFirst=['w2v_tag_first'+str(i) for i in range(32)]
                columnsW2vTagFirst.append(action)
                g=tmp.groupby('userid')[columnsW2vTagFirst].apply(lambda x:getEmbed(x,action))
                g=g.drop(columns=action)
                columnsW2vTagFirst.remove(action)
                columnsUserW2vTagFirst=['userid_'+i+'_'+action for i in columnsW2vTagFirst]
                g.columns=columnsUserW2vTagFirst
                g.reset_index(level=0, inplace=True)
                g['date_']=tmp['date_']
                g = g[stat_cols + columnsUserW2vTagFirst + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)#去重
                stat_df = pd.concat([stat_df, g], axis=0, ignore_index=True)
                
                del g, tmp
            stat_df.to_csv(FEATURE_PATH+'key_tag_{}.csv'.format(action),index=False)
            dataDF = dataDF.merge(stat_df, on=stat_cols + ['date_'], how='left')
            del stat_df
            gc.collect()

    dataDF = reduce_mem(dataDF,list(dataDF.columns))

    """
    计算相似度
    """
    for action in tqdm(ACTION_LIST[:4]):
        stat_df=pd.read_csv(FEATURE_PATH+'key_tag_{}.csv'.format(action))
        stat_df = stat_df.drop_duplicates(['date_','date_']).reset_index(drop=True)#去重
        dataDF = dataDF.merge(stat_df, on=['userid' ,'date_'], how='left')
        columnsW2vKeywordFirst=['w2v_keyword_first'+str(i) for i in range(32)]
        columnsUserW2vKeywordFirst=['userid_'+i+'_'+action for i in columnsW2vKeywordFirst]
        dataDF['cosSimW2vKeywordFirst_'+action]=(np.sum(dataDF[columnsW2vKeywordFirst].values*dataDF[columnsUserW2vKeywordFirst].values,axis=1))/(np.linalg.norm(dataDF[columnsW2vKeywordFirst].values,axis=1)*np.linalg.norm(dataDF[columnsUserW2vKeywordFirst].values,axis=1))
        
        columnsW2vTagFirst=['w2v_tag_first'+str(i) for i in range(32)]
        columnsUserW2vTagFirst=['userid_'+i+'_'+action for i in columnsW2vTagFirst]
        dataDF['cosSimW2vTagFirst_'+action]=(np.sum(dataDF[columnsW2vTagFirst].values*dataDF[columnsUserW2vTagFirst].values,axis=1))/(np.linalg.norm(dataDF[columnsW2vTagFirst].values,axis=1)*np.linalg.norm(dataDF[columnsUserW2vTagFirst].values,axis=1))
        remove_columns=[i for i in stat_df.columns if i not in ['userid','date_']]
        dataDF=dataDF.drop(columns=remove_columns)
        
        np.save(FEATURE_PATH+"cosSimW2vKeywordFirst_{}.npy".format(action), dataDF['cosSimW2vKeywordFirst_'+action].values)
        np.save(FEATURE_PATH+"cosSimW2vTagFirst_{}.npy".format(action), dataDF['cosSimW2vTagFirst_'+action].values)
        del stat_df
        gc.collect()

    # columnsW2vKeywordFirst=['w2v_keyword_first'+str(i) for i in range(32)]
    # columnsW2vTagFirst=['w2v_tag_first'+str(i) for i in range(32)]
    # columnsUserW2vKeywordFirst=['userid_'+i+'_'+action for i in columnsW2vKeywordFirst for action in ACTION_LIST[:4]]
    # columnsUserW2vTagFirst=['userid_'+i+'_'+action for i in columnsW2vTagFirst  for action in ACTION_LIST[:4]]
    columnsSimKeywordFirst=['cosSimW2vKeywordFirst_'+action for action in ACTION_LIST[:4]]
    columnsSimTagFirst=['cosSimW2vTagFirst_'+action for action in ACTION_LIST[:4]]

    # columns=columnsW2vKeywordFirst+columnsW2vTagFirst+columnsUserW2vKeywordFirst+columnsUserW2vTagFirst+columnsSimKeywordFirst+columnsSimTagFirst
    columns=columnsSimKeywordFirst+columnsSimTagFirst
    dataDF[columns].to_pickle(FEATURE_PATH+'dataDF_keyword_tag_w2v_testb.pkl')

def getTagSim(dataDF):
    feedDF = pd.read_csv(FEED_INFO_PATH)
    # dataDF['machine_tag_list'].fillna('-1', inplace = True)
    dataDF['tag_maxprob'] = list(map(find_tag, dataDF["machine_tag_list"].values))
    feedDF['tag_maxprob'] = list(map(find_tag, feedDF["machine_tag_list"].values))

    txt1=dataDF['machine_tag_list'].apply(lambda x:getTextMachine(x))
    txt2=dataDF['manual_tag_list'].apply(lambda x:getText(x))
    txt1=list(txt1)
    txt2=list(txt2)
    txt1.extend(txt2)

    model = Word2Vec(txt1, sg=1, size=32,  window=5,  min_count=2,  negative=3, sample=0.001, hs=1, workers=32)
    # 保存模型
    model.save(MODEL_PATH+"tagWord2vec_testab.model")
        
    userVec=dict()
    # stat_df=pd.DataFrame()
    for target_day in tqdm(range(2, MAX_DAY + 1)):
        # left, right = max(target_day - n_day, 1), target_day - 1 #[left,right]是target_day前n天的天，[1,1]:1\\[1,2]:2\\...\\[1,5]:6\\...
        tmp = dataDF[dataDF['date_'] <= target_day-1].reset_index(drop=True) # 取出target_day前的数据
        tmp['date_'] = target_day

        t=tmp.groupby('userid')[['tag_maxprob']+ACTION_LIST[:4]].apply(lambda x:getEmbed(x,ACTION_LIST[:4],model))
        vec={}
        vecM=t.values
        for idx,u in enumerate(t.index):
            vec[u]=vecM[idx]
        userVec[target_day]=vec
        del t, tmp
    dataDF['sim_w2v_tag_maxprob_userid']=dataDF[['userid','date_','tag_maxprob']].apply(lambda x:calSim(x[0],x[1],x[2],userVec,model),axis=1)
    np.save(FEATURE_PATH+"sim_w2v_tag_maxprob_userid_testab_all.npy", dataDF['sim_w2v_tag_maxprob_userid'].values)
    np.save(FEATURE_PATH+"sim_w2v_tag_maxprob_userid_testab.npy", dataDF.iloc[:uselen]['sim_w2v_tag_maxprob_userid'].values)
    return dataDF

def getGraphEmbedSim(dataDF,uselen):
    graphEmbedding=np.load(FEATURE_PATH+"grap_embedding33_bfeedid.npy")
    feedDF = pd.read_csv(FEED_INFO_PATH)
    feedidGraphEmbed=dict()
    for idx,feed in enumerate(feedDF['feedid']):
        feedidGraphEmbed[str(feed)]=graphEmbedding[idx]
    """
    用w2v计算userid的w2v
    """

    # n_day=5

    userVec=dict()
    # stat_df=pd.DataFrame()
    for target_day in tqdm(range(2, MAX_DAY + 1)):
        # left, right = max(target_day - n_day, 1), target_day - 1 #[left,right]是target_day前n天的天，[1,1]:1\\[1,2]:2\\...\\[1,5]:6\\...
        tmp = dataDF[dataDF['date_'] <= target_day-1].reset_index(drop=True) # 取出target_day前的数据
        tmp['date_'] = target_day

        t=tmp.groupby('userid')[['feedid']+ACTION_LIST[:4]].apply(lambda x:getEmbed(x,ACTION_LIST[:4],feedidGraphEmbed))
        vec={}
        vecM=t.values
        for idx,u in enumerate(t.index):
            vec[u]=vecM[idx]
        userVec[target_day]=vec
        del t, tmp
    dataDF['sim_graphembed_feedid_userid']=dataDF[['userid','date_','feedid']].apply(lambda x:calSim1(x[0],x[1],x[2],userVec,feedidGraphEmbed),axis=1)
    np.save(FEATURE_PATH+"sim_graphEmbed_feedid_userid_testab_all.npy", dataDF['sim_graphembed_feedid_userid'].values)
    np.save(FEATURE_PATH+"sim_graphEmbed_feedid_userid_testab.npy", dataDF.iloc[:uselen]['sim_graphembed_feedid_userid'].values)

def getFeatureRank(dataDF,uselen):
    """
    相似度rank以及attention值
    """
    for action in tqdm(ACTION_LIST[:4]):
    #     dataDF['cosSimW2vTagFirst_'+action+'_rank']=dataDF.groupby(['userid','date_'])['cosSimW2vTagFirst_'+action].transform('rank')
    #     dataDF['cosSimW2vKeywordFirst_'+action+'_rank']=dataDF.groupby(['userid','date_'])['cosSimW2vKeywordFirst_'+action].transform('rank')
        dataDF['cosSimEmbed_'+action+'_rank']=dataDF.groupby(['userid','date_'])['cosSimEmbed_'+action].transform('rank')
    #     dataDF['cosSimW2v_'+action+'_rank']=dataDF.groupby(['userid','date_'])['cosSimW2v_'+action].transform('rank')
        
        tmp=dataDF[['userid','date_','cosSimEmbed_'+action]].copy()
        tmp['cosSimEmbed_'+action]=np.exp(tmp['cosSimEmbed_'+action])
    #     tmp['cosSimW2v_'+action]=np.exp(tmp['cosSimW2v_'+action])
        dataDF['cosSimEmbed_'+action+'_attention']=tmp['cosSimEmbed_'+action]/tmp.groupby(['userid','date_'])['cosSimEmbed_'+action].transform('sum')
    #     dataDF['cosSimW2v_'+action+'_attention']=tmp['cosSimW2v_'+action]/tmp.groupby(['userid','date_'])['cosSimW2v_'+action].transform('sum')
    #     dataDF['cosSimW2vTagFirst_'+action+'_attention']=np.exp(dataDF['cosSimW2vTagFirst_'+action])/np.exp(dataDF.groupby('userid')['cosSimW2vTagFirst_'+action]).transform('sum')
    #     dataDF['cosSimW2vKeywordFirst_'+action+'_attention']=np.exp(dataDF['cosSimW2vKeywordFirst_'+action])/np.exp(dataDF.groupby('userid')['cosSimW2vKeywordFirst_'+action]).transform('sum')

    tmp=dataDF[['userid','date_','sim_graphEmbed_feedid_userid','sim_w2v_tag_maxprob_userid']].copy()
    tmp['sim_graphEmbed_feedid_userid']=np.exp(tmp['sim_graphEmbed_feedid_userid'])
    tmp['sim_w2v_tag_maxprob_userid']=np.exp(tmp['sim_w2v_tag_maxprob_userid'])

    dataDF['sim_graphEmbed_feedid_userid_rank']=dataDF.groupby(['userid','date_'])['sim_graphEmbed_feedid_userid'].transform('rank')
    dataDF['sim_graphEmbed_feedid_userid_att']=tmp['sim_graphEmbed_feedid_userid']/tmp.groupby(['userid','date_'])['sim_graphEmbed_feedid_userid'].transform('sum')

    dataDF['sim_w2v_tag_maxprob_userid_rank']=dataDF.groupby(['userid','date_'])['sim_w2v_tag_maxprob_userid'].transform('rank')
    dataDF['sim_w2v_tag_maxprob_userid_att']=tmp['sim_w2v_tag_maxprob_userid']/tmp.groupby(['userid','date_'])['sim_w2v_tag_maxprob_userid'].transform('sum')

    columnsCosSimEmbedRank=['cosSimEmbed_'+action+'_rank' for action in ACTION_LIST[:4]]
    # columnsCosSimKeywordFirstRank=['cosSimW2vKeywordFirst_'+action+'_rank' for action in ACTION_LIST[:4]]
    # columnsCosSimTagFirstRank=['cosSimW2vTagFirst_'+action+'_rank' for action in ACTION_LIST[:4]]
    # columnsCosSimW2vRank=['cosSimW2v_'+action+'_rank' for action in ACTION_LIST[:4]]
    
    columnsCosSimEmbedAtt=['cosSimEmbed_'+action+'_attention' for action in ACTION_LIST[:4]]

    columnsadd=['sim_graphEmbed_feedid_userid_rank','sim_graphEmbed_feedid_userid_att','sim_w2v_tag_maxprob_userid_rank','sim_w2v_tag_maxprob_userid_att']
    # columnsCosSimKeywordFirstAtt=['cosSimW2vKeywordFirst_'+action+'_attention' for action in ACTION_LIST[:4]]
    # columnsCosSimTagFirstAtt=['cosSimW2vTagFirst_'+action+'_attention' for action in ACTION_LIST[:4]]
    # columnsCosSimW2vAtt=['cosSimW2v_'+action+'_attention' for action in ACTION_LIST[:4]]
    columns=columnsCosSimEmbedRank+columnsCosSimEmbedAtt+columnsadd#+columnsCosSimKeywordFirstRank+columnsCosSimTagFirstRank+columnsCosSimKeywordFirstAtt+columnsCosSimTagFirstAtt
    dataDF[columns].to_pickle(FEATURE_PATH+'dataDF_rank_testab_all.pkl')
    dataDF.iloc[:uselen][columns].to_pickle(FEATURE_PATH+'dataDF_rank_testab.pkl')



def getFeatureGraphEmbed(dataDF):
    feedDF = pd.read_csv(FEED_INFO_PATH)
    graphEmbedding=np.load(FEATURE_PATH+"grap_embedding33_bfeedid.npy")
    graphEmbedDF=pd.DataFrame(graphEmbedding,columns=['graphEmbed'+str(i) for i in range(32)])
    feedDF=pd.concat([feedDF,graphEmbedDF],axis=1)
    dataDF = dataDF.merge(feedDF, on='feedid', how='left')
    dataDF[['graphEmbed'+str(i) for i in range(32)]].to_pickle(FEATURE_PATH+"graphEmbedFeature_testab.pkl")
    return dataDF

def getFeature(dataDFAll,uselen):
    dataDFAll=getFeatureTagKeyword(dataDFAll,uselen)
    getFeatureStatistic(dataDFAll,uselen)
    getFeatureWindows(dataDFAll.iloc[:uselen],3)
    getFeatureWindows(dataDFAll.iloc[:uselen],5)
    getFeatureWindows(dataDFAll.iloc[:uselen],7)
    getFeatureGraphEmbed(dataDFAll.iloc[:uselen])
    dataDFAll=getFeatureEmbedding(dataDFAll,uselen)
    dataDFAll=getTagSim(dataDFAll,uselen)
    dataDFAll=getGraphEmbedSim(dataDFAll,uselen)
    getFeatureRank(dataDFAll,uselen)

def mergeFeatureTotal(dataDF):
    """
    合并部分特征
    """
    # # ##统计特征
    dataStatistic=pd.read_pickle(FEATURE_PATH+'statisticFeature15_testab.pkl')
    dataStatisticFeatures=list(dataStatistic.columns)
    dataDF=pd.concat([dataDF,dataStatistic],axis=1)
    del dataStatistic

    # ## tag和keyword生成的特征
    dataTagKeyword=pd.read_pickle(FEATURE_PATH+'tag_keyword_features_testab.pkl')
    dataTagKeywordFeatures=list(dataTagKeyword.columns)
    dataDF=pd.concat([dataDF,dataTagKeyword],axis=1)
    del dataTagKeyword

    ## 滑动窗口特征
    dataWindow_5=pd.read_pickle(FEATURE_PATH+'windowsFeature5day_testb.pkl')
    dataWindowADD1=pd.read_pickle(FEATURE_PATH+'windowsFeature5dayNew1_testb.pkl')
    # dataWindowADD2=pd.read_pickle(FEATURE_PATH+'windowsFeatureNew2.pkl')
    # dataWindowADD3=pd.read_pickle(FEATURE_PATH+'windowsFeatureNew3.pkl')
    # dataWindowADD4=pd.read_pickle(FEATURE_PATH+'windowsFeatureNew4.pkl')
    # dataWindowADD5=pd.read_pickle(FEATURE_PATH+'windowsFeatureNew5.pkl')

    dataWindow_3=pd.read_pickle(FEATURE_PATH+'windowsFeature3day_testb.pkl')
    dataWindowADD1_3=pd.read_pickle(FEATURE_PATH+'windowsFeature3dayNew1_testb.pkl')
    # dataWindowADD2_3=pd.read_pickle(FEATURE_PATH+'windowsFeature3dayNew2.pkl')
    # dataWindowADD3_3=pd.read_pickle(FEATURE_PATH+'windowsFeature3dayNew3.pkl')
    # dataWindowADD4_3=pd.read_pickle(FEATURE_PATH+'windowsFeature3dayNew4.pkl')
    # dataWindowADD5_3=pd.read_pickle(FEATURE_PATH+'windowsFeature3dayNew5.pkl')

    dataWindow_7=pd.read_pickle(FEATURE_PATH+'windowsFeature7day_testb.pkl')
    dataWindowADD1_7=pd.read_pickle(FEATURE_PATH+'windowsFeature7dayNew1_testb.pkl')
    # dataWindowADD2_7=pd.read_pickle(FEATURE_PATH+'windowsFeature7dayNew2.pkl')
    # dataWindowADD3_7=pd.read_pickle(FEATURE_PATH+'windowsFeature7dayNew3.pkl')
    # dataWindowADD4_7=pd.read_pickle(FEATURE_PATH+'windowsFeature7dayNew4.pkl')
    # dataWindowADD5_7=pd.read_pickle(FEATURE_PATH+'windowsFeature7dayNew5.pkl')

    dataWindow=pd.concat([dataWindow_5,dataWindowADD1,#dataWindowADD2,dataWindowADD3,dataWindowADD4,dataWindowADD5,
                        dataWindow_3,dataWindow_7,dataWindowADD1_3,dataWindowADD1_7
    #                      dataWindow_3,dataWindowADD1_3,dataWindowADD2_3,dataWindowADD3_3,dataWindowADD4_3,dataWindowADD5_3,
    #                      dataWindow_7,dataWindowADD1_7,dataWindowADD2_7,dataWindowADD3_7,dataWindowADD4_7,dataWindowADD5_7,
                        ],axis=1)
    dataWindowFeatures=list(dataWindow.columns)
    dataDF=pd.concat([dataDF,dataWindow],axis=1)

    del dataWindow_5,dataWindowADD1#,dataWindowADD2,dataWindowADD3,dataWindowADD4,dataWindowADD5,
    del dataWindow_3,dataWindow_7,dataWindowADD1_3,dataWindowADD1_7
    del dataWindow
    # del dataWindow_3,dataWindowADD1_3,dataWindowADD2_3,dataWindowADD3_3,dataWindowADD4_3,dataWindowADD5_3
    # del dataWindow_7,dataWindowADD1_7,dataWindowADD2_7,dataWindowADD3_7,dataWindowADD4_7,dataWindowADD5_7

    ##Embedding PCA降维特征
    dataEmbed=pd.read_pickle(FEATURE_PATH+'featureEmbedding_testab.pkl')
    dataEmbedFeature=dataEmbed.columns
    dataDF=pd.concat([dataDF,dataEmbed[dataEmbedFeature[-4:]]],axis=1) #只取相似度
    del dataEmbed

    # ##feedid word2vec特征
    # dataW2vFeedid=pd.read_pickle(FEATURE_PATH+'featureW2vFeedid.pkl')
    # dataW2vFeedidFeature=dataW2vFeedid.columns
    # dataDF=pd.concat([dataDF,dataW2vFeedid[dataW2vFeedidFeature[-4:]]],axis=1)
    # del dataW2vFeedid

    ## 当天的排序特征
    ## 可能要去掉，线上检验一下这个的问题。
    dataRank=pd.read_pickle(FEATURE_PATH+'dataDF_rank_testab.pkl')
    dataRankFeature=dataRank.columns
    dataDF=pd.concat([dataDF,dataRank],axis=1)
    del dataRank

    ## graph embedding
    dataGraphEmbed=pd.read_pickle(FEATURE_PATH+'graphEmbedFeature_testab.pkl')
    dataGraphEmbedFeature=dataGraphEmbed.columns
    dataDF=pd.concat([dataDF,dataGraphEmbed],axis=1)
    del dataGraphEmbed

    sim_graphEmbed_feedid_userid=np.load(FEATURE_PATH+'sim_graphEmbed_feedid_userid_testab.npy')
    sim_w2v_tag_maxprob_userid=np.load(FEATURE_PATH+'sim_w2v_tag_maxprob_userid_testab.npy')
    dataDF['sim_graphEmbed_feedid_userid']=sim_graphEmbed_feedid_userid
    dataDF['sim_w2v_tag_maxprob_userid']=sim_w2v_tag_maxprob_userid
    
    play_cols = ['is_finish', 'play_times','play_stay', 'play', 'stay','stay_minus_play']
    KEYTAG=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list','description', 'ocr', 'asr']
    dataDF = reduce_mem(dataDF, list(dataDF.columns))
    dataDF.to_pickle(FEATURE_PATH+'dataDFtestab.pkl')
    return dataDF

def calFeatureImportance():
    """
    划分数据集
    """
    dataDF=pd.read_pickle(FEATURE_PATH+'dataDFtestab.pkl')
    play_cols = ['is_finish', 'play_times','play_stay', 'play', 'stay','stay_minus_play']
    KEYTAG=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list','description', 'ocr', 'asr']
    cols = [f for f in dataDF.columns if f not in ['date_'] + play_cols + ACTION_LIST+KEYTAG]
    dataDF = reduce_mem(dataDF, [f for f in dataDF.columns if f not in ['date_'] + play_cols + ACTION_LIST])

    train = dataDF[~dataDF['read_comment'].isna()].reset_index(drop=True)
    # test = dataDF[dataDF['read_comment'].isna()].reset_index(drop=True)

    trn_x = train[train['date_'] < 14].reset_index(drop=True)
    val_x = train[train['date_'] == 14].reset_index(drop=True)
    # dataDF.to_pickle(FEATURE_PATH+'dataDF{}.pkl'.format(892))
    del dataDF,train
    gc.collect()

    ##################### 线下验证 #####################
    uauc_list = []
    r_list = []
    for idx,y in enumerate(ACTION_LIST[:4]):
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            is_unbalance = True,
            learning_rate=0.05,
            n_estimators=5000,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021,
            metric='None'
        )
        
        clf.fit(
            trn_x[cols], trn_x[y],
            eval_set=[(val_x[cols], val_x[y])],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50
        )
        importanceDF=pd.DataFrame({
                    'column': clf.booster_.feature_name(),
                    'importance': clf.booster_.feature_importance(),
                }).sort_values(by='importance',ascending=False)
        importanceDF.to_csv(FEATURE_PATH+'importanceDF_val_{}_testab.csv'.format(y),index=False)

        val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]
        val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])
        uauc_list.append(val_uauc)
        print(val_uauc)
        r_list.append(clf.best_iteration_)
        print('runtime: {}\n'.format(time.time() - t))
    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
    print(uauc_list)
    print(weighted_uauc)


if __name__=='__main__':
    dataDFAll,uselen=readData(readTestA=True)
    getFeature(dataDFAll,uselen)
    mergeFeatureTotal(dataDFAll.iloc[:uselen])
    calFeatureImportance()
    