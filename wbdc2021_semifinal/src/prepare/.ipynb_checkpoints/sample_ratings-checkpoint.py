import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import pickle
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

ROOT_PATH='../../data/'
ratings=pd.read_csv(ROOT_PATH+'wedata/wechat_algo_data2/user_action.csv')
PREDICT_LIST=["read_comment","like", "click_avatar", "forward",'comment','follow','favorite']
ratings=ratings.drop_duplicates(['userid', 'feedid'], keep='last')
print(ratings[PREDICT_LIST].mean(0))
# 采样频率
ACTION_SAMPLE_RATE={"read_comment":2.5,"like":2.5, "click_avatar":6, "forward":10,'comment':30,'follow':20,'favorite':17}
df_all=pd.DataFrame()
for action in tqdm(PREDICT_LIST):
    df_neg = ratings[ratings[action] == 0]
    df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
    df_all = pd.concat([df_all,df_neg, ratings[ratings[action] == 1]])
    df_all=df_all.drop_duplicates(['userid', 'feedid'], keep='last')

print(df_all[PREDICT_LIST].mean(0)) 
print('-------采样结束------------------')
df=df_all
del ratings
gc.collect()
feed_info=pd.read_pickle(ROOT_PATH+'tmp/cat_feed_info.pkl')
#----------------------------
df = df.merge(feed_info[['feedid', 'authorid', 'videoplayseconds','bgm_song_id','manual_keyword_id1',]], on='feedid', how='left')
## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000
df[df['play']>240000]=240000
df['is_finish'] = (df['play'] >= df['videoplayseconds']*0.92).astype('int8')
# df['play_times'] = (df['play'] / df['videoplayseconds']).astype('float16')
play_cols = ['is_finish']
df=reduce_mem(df)
gc.collect()
n_day =12
#--------------------加载交互特征--------------------
max_day=14
for stat_cols in tqdm([  ['userid'],['feedid'],['authorid'], ['userid', 'authorid'],['userid', 'bgm_song_id'],
        ['userid','manual_keyword_id1']]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
#         tmp.to_pickle('./tmp/{}_feat_{}.pkl'.format(target_day,'_'.join(stat_cols)))
        tmp=pd.read_pickle(ROOT_PATH+'tmp/{}_feat_{}.pkl'.format(target_day,'_'.join(stat_cols)))
        tmp=reduce_mem(tmp)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del tmp
    mean_tmp=pickle.load(open(ROOT_PATH+'tmp/{}_feat_mean.pkl'.format('_'.join(stat_cols)),'rb'))
#     pickle.dump(mean_tmp,open(ROOT_PATH+'tmp/{}_feat_mean.pkl'.format('_'.join(stat_cols)),'wb'))# 保存填充nan的均值
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    for kk,vv in mean_tmp.items():
        df[kk]=df[kk].fillna(vv) # 填充均值
    df=reduce_mem(df)
    del stat_df
    gc.collect()
df.fillna(-1,inplace=True)
feat=pickle.load(open(ROOT_PATH+'tmp/feat_list.pkl','rb')) # 保存特征列表
print(len(feat))

normolizer_dict=pickle.load(open(ROOT_PATH+'tmp/normolizer_dict.pkl','rb'))
for f in tqdm(feat):
    tmp=df[f].values.astype('float16').clip(-1,1e8)
    tmp_max=normolizer_dict[f+'_max'] # 这里 或许我得保留均值和方差
    tmp_min=normolizer_dict[f+'_min']
    df[f]=((tmp-tmp_min)/tmp_max).astype('float16')   
df['reg']=np.sqrt((df['play']/df['videoplayseconds']).values)   
df.to_pickle(ROOT_PATH+'tmp/sample_ratings_feat_df.pkl')