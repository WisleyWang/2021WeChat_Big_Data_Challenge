ROOT_PATH = "../data" # 存储数据的根目录
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/' # 比赛数据集路径
SUBMIT_PATH = ROOT_PATH + '/submission/' # 比赛数据集路径
MODEL_PATH = ROOT_PATH + '/model/' # 比赛数据集路径
FEATURE_PATH = ROOT_PATH + '/tmp/' # 比赛数据集路径

# 训练集
USER_ACTION_PATH = DATASET_PATH + "user_action.csv"
FEED_INFO_PATH = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS_PATH = DATASET_PATH + "feed_embeddings.csv"

# 测试集
TEST_FILE_PATH = DATASET_PATH + "test_b.csv"
TESTA_FILE_PATH=DATASET_PATH + "test_a.csv"


# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_USER_LIST = ["device","play","stay","read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]

FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
KEYTAG=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']

MAX_DAY = 15

SEED_LIST=[3,12,57,2021,100]