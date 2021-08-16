# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm


# 存储数据的根目录
ROOT_PATH = "../data/"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
SUBMIT_PATH = ROOT_PATH + '/submission/' 
MODEL_PATH = ROOT_PATH + '/model/'
FEATURE_PATH = ROOT_PATH + '/tmp/' 
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = FEATURE_PATH + "embedPCA32.csv"
GRAPH_EMBEDDING = FEATURE_PATH + "grap_embedding32_feedid_b1.npy"
# 测试集
TEST_FILE = DATASET_PATH + "test_b.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
#ACTION_LIST = ["forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite",'device']
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'description', 'ocr', 'asr',
 'description_char', 'ocr_char', 'asr_char', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list',
                'machine_tag_list', '0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
       '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
       '30', '31']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}

def process_embed(train):
    feed_embed_array = np.zeros((train.shape[0], 512))
    for i in tqdm(range(train.shape[0])):
        x = train.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    train = pd.concat((train, temp), axis=1)
    return train

def prepare_data():
    feed_info_df = pd.read_csv(FEED_INFO)
    #print(feed_info_df.columns)
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    #print(feed_embed.columns)
    test = pd.read_csv(TEST_FILE)

    ## process machine_tag_list
    def find_tag(x):
        if x == '-1':
            return x
        max_prob = 0
        for i in x.split(';'):
            key = int(i.split(' ')[0])
            prob = float(i.split(' ')[1])
            if prob >= max_prob:
                max_prob = prob
                max_key = key
        return max_key

    feed_info_df['machine_tag_list'].fillna('-1', inplace = True)
    genres_list = list(map(find_tag, feed_info_df['machine_tag_list'].values))
    feed_info_df['machine_tag_list'] = genres_list
    
    ##add graph embedding
    graph_embedding = np.load(GRAPH_EMBEDDING)
    graph_embedding = pd.DataFrame(graph_embedding)
    feed_info_df = pd.concat([feed_info_df,graph_embedding], axis = 1)

    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')

    train = pd.merge(train, feed_embed, on='feedid', how='left')
    test = pd.merge(test, feed_embed, on='feedid', how='left')

    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
    
    test.to_csv(FEATURE_PATH + f'/test_data.csv', index=False) 
    for action in tqdm(ACTION_LIST):
        print(f"prepare data for {action}")
        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=9, replace=False)
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        print(df_all.columns)
        df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
        df_all.to_csv(FEATURE_PATH + f'/train_data_for_{action}.csv', index=False)


if __name__ == "__main__":
    prepare_data()
