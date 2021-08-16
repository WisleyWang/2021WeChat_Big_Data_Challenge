'''将单个模型的预测合并'''

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))

DATA_PATH=os.path.join(BASE_DIR, '../data/wedata/wechat_algo_data1/')
SUBMMIT_PATH=os.path.join(BASE_DIR, '../data/submission/')
# 读取--
test_b=pd.read_csv(os.path.join(DATA_PATH,'test_b.csv'))
PREDICT_LIST=["read_comment","like", "click_avatar", "forward"]

sub1=pd.read_csv(os.path.join(SUBMMIT_PATH,'my_deep_v2_v1_b1.csv'))
sub2=pd.read_csv(os.path.join(SUBMMIT_PATH,'my_deep_v2_v2_b1.csv'))
# 加权--
test_pred=sub1[PREDICT_LIST].values*0.5+sub2[PREDICT_LIST].values*0.5

# sub
sub=test_b[['userid','feedid']]
for i in range(4):
    sub[PREDICT_LIST[i]]=test_pred[:,i]

sub.to_csv(os.path.join(SUBMMIT_PATH,'nn_finall_b1.csv'),index=False)

## sub of lgb
sub_lgb_1=pd.read_csv(os.path.join(SUBMMIT_PATH,'sub_lgb_testab_seed3.csv'))
sub_lgb_2=pd.read_csv(os.path.join(SUBMMIT_PATH,'sub_lgb_testab_seed12.csv'))
sub_lgb_3=pd.read_csv(os.path.join(SUBMMIT_PATH,'sub_lgb_testab_seed57.csv'))
sub_lgb_4=pd.read_csv(os.path.join(SUBMMIT_PATH,'sub_lgb_testab_seed2021.csv'))
sub_lgb_5=pd.read_csv(os.path.join(SUBMMIT_PATH,'sub_lgb_testab_seed100.csv'))
sub_lgb_1[PREDICT_LIST]=0.2*sub_lgb_1[PREDICT_LIST]+0.2*sub_lgb_2[PREDICT_LIST]+0.2*sub_lgb_3[PREDICT_LIST]+0.2*sub_lgb_4[PREDICT_LIST]+0.2*sub_lgb_5[PREDICT_LIST]
sub_lgb_1.to_csv(os.path.join(SUBMMIT_PATH,'lgb_finall_b1.csv'),index=False)



## sub of kitty
sub0=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_deepfm_0.csv'))
sub1=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_deepfm_1.csv'))
sub2=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_deepfm_2.csv'))
sub3=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_deepfm_3.csv'))
sub4=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_deepfm_4.csv'))
sub5=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_deepfm_5.csv'))

sub6=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_autoint_0.csv'))
sub7=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_autoint_1.csv'))
sub8=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_autoint_2.csv'))
sub9=pd.read_csv(os.path.join(SUBMMIT_PATH,'submit_b_autoint_3.csv'))

sub0[PREDICT_LIST]=(sub0[PREDICT_LIST]+sub1[PREDICT_LIST]+sub2[PREDICT_LIST]+sub3[PREDICT_LIST]+
                    sub4[PREDICT_LIST]+sub5[PREDICT_LIST]+sub6[PREDICT_LIST]+sub7[PREDICT_LIST]+
                    sub8[PREDICT_LIST]+sub9[PREDICT_LIST])/10
sub0.to_csv(os.path.join(SUBMMIT_PATH,'deepfm_finall_b1.csv'),index=False)
##Total merge

sub['read_comment']=0.35*sub['read_comment']+0.1*sub_lgb_1['read_comment']+0.55*sub0['read_comment']
sub['like']=0.15*sub['like']+0.7*sub_lgb_1['like']+0.15*sub0['like']
sub['click_avatar']=0.4*sub['click_avatar']+0.3*sub_lgb_1['click_avatar']+0.3*sub0['click_avatar']
sub['forward']=0.35*sub['forward']+0.4*sub_lgb_1['forward']+0.25*sub0['forward']

sub.to_csv(os.path.join(SUBMMIT_PATH,'result.csv'),index=False)