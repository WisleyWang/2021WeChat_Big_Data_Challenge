from configLightgbm import *
import joblib
import pandas as pd
import gc

if __name__=='__main__':
    cols_add=[]#['sim_graphEmbed_feedid_userid','sim_w2v_tag_maxprob_userid'] #除了重要性之外其他特征
    cols_list=[]
    cols_total=[]
    for y in ACTION_LIST[:4]:
        cols_y=[]
        importanceDF=pd.read_csv(FEATURE_PATH+'importanceDF_val_{}_testab.csv'.format(y))
        cols_y=list(importanceDF['column'][:300])
        cols_y.extend(cols_add)
        cols_list.append(cols_y)
        cols_total.extend(cols_y)
    cols_total=set(cols_total)

    """
    划分数据集
    """
    dataDF=pd.read_pickle(FEATURE_PATH+'dataDFtestab.pkl')
    play_cols = ['is_finish', 'play_times','play_stay', 'play', 'stay','stay_minus_play']
    KEYTAG=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list',
                    'description', 'ocr', 'asr']
    # cols = [f for f in dataDF.columns if f not in ['date_'] + play_cols + ACTION_LIST+KEYTAG]
    # dataDF = reduce_mem(dataDF, [f for f in dataDF.columns if f not in ['date_'] + play_cols + ACTION_LIST])

    train = dataDF[~dataDF['read_comment'].isna()].reset_index(drop=True)
    test = dataDF[dataDF['read_comment'].isna()].reset_index(drop=True)

    # trn_x = train[train['date_'] < 14].reset_index(drop=True)
    # val_x = train[train['date_'] == 14].reset_index(drop=True)
    # dataDF.to_pickle(FEATURE_PATH+'dataDF{}.pkl'.format(892))
    del dataDF
    gc.collect()

    for seed in SEED_LIST:
        for idx,y in enumerate(ACTION_LIST[:4]):
            print('=========', y, '=========')
            # 模型加载
            clf = joblib.load(MODEL_PATH+'lgb_model_testab_{}_seed{}.pkl'.format(y,seed))
            test[y] = clf.predict_proba(test[cols_list[idx]])[:, 1]
        
        test[['userid', 'feedid'] + ACTION_LIST[:4]].to_csv(
            SUBMIT_PATH+'sub_lgb_testab_seed{}.csv'.format(seed),
            index=False
        )   
