# 2021微信大数据挑战赛【初赛】
本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。

本次比赛以多个行为预测结果的加权uAUC值进行评分。大赛官方网站：https://algo.weixin.qq.com/
## 介绍
本次代码nn部分主要采用deepctr框架，torch及tensorflow版本均有.

成员工作：
- Kitty: 负责deepctr模型(deepfm、autoint)
- 近西: 负责lightgbm模型
- 万金油： 负责gnn、ctr等模型搭建
### 1、文件夹目录结构
```
./
|-- init.sh # 环境初始化sh
|-- train.sh # 主执sh
|-- README.md # 解决方案及算法介绍
|-- requirements.txt # python环境依赖
|-- src # 代码存放
    |-- my_deep_v2_v1.py # 训练+预测 model1
    |-- my_deep_v2_v2.py # 训练+预测 model2
    |-- prepare.py # 数据处理,特征生成
    |-- inference.py # 推断,融合
    |-- configLightgbm.py # lgb参数
    |-- prepareFeatureLightgbm.py # 生成lgb特征
    |-- tools.py # 工具函数
    |-- trainLightgbm.py # 训练lgb
    |-- inferenceLightgbm.py # lgb推断
    |-- prepare_data_for_deepfm.py # 生成deepfm特征
    |-- DeepFM.py # 训练+预测 deepfm 
    |-- AutoInt.py # 训练+预测 autoint
|-- data  #数据存放
	|-- model  # 存放模型
	|-- submission # 存放预测文件及最终提交文件
	|-- tmp # 存放 特征,预处理等临时文件
	|-- wedata  # 存放比赛数据
        |-- wechat_algo_data1
```
### 2、环境配置
初赛采用nn+lgb模型，
nn模型包括特征提取平均每个模型训练耗时:1小时。
lgb模型包括特征提取平均每个模型训练耗时:8小时。
- pandas==1.2.4
- tensorflow==2.4.1
- numpy==1.19.5
- scikit_learn==0.24.2
- Keras==2.4.3
- gensim==3.8.3
- deepctr-torch==0.2.6
- deepctr==0.8.5
- dgl==0.6.1
- torch==1.8.1
- tqdm==4.59.0
- sklearn
- numpy==1.19.5

### 3. 运行配置
- GPU 
- 最小内存要求
  - 特征/样本生成 ：192G
  - 模型训练及评估：128G
- 耗时
    - 测试环境：内存192G，CPU 2.3 GHz 双核Intel Core i5
    - 特征/样本生成：8 h
    - 模型训练及评估：4 h
- 预处理

### 4. 运行流程

安装环境：sh init.sh
将初赛数据集解压后放到data/wedata目录下，得到data/wedata/wechat_algo_data1
数据准备和模型训练,预测：sh train.sh
融合生成结果文件：sh inference.sh

### 5、模型结果

| 模型 | 线上A榜得分 |线上B榜得分|查看评论 |点赞|点击头像|转发|
| :---:  | :-----:| :-----:| :-----:| :-----:|:-----:|:-----:|
| lgb  | 0.667 | | 0.636 |0.6505 |0.7405 |0.6903 |
|nn|0.669504| |0.644638|0.646725|0.73873|0.698853|
| lgb+nn |   |0.67632|0.6493|0.6510|0.7453| 0.7217|

### 6、 特征说明
lightgbm特征生成了884个，进行特征筛选后降低到300个。

主要包括统计特征、Tag、keyword的第一个值与概率最大的值、前3、5、7天的滑动窗口特征、feedid计算的word2vec特征、tag计算的word2vec特征，feedid的embedding，以及通过feedid端的嵌入计算userid的嵌入，并计算feedid和userid的余弦相似度。

DeepFM和AutoInt模型主要使用了数据本身的id，description, ocr, asr, tag, keyword, feed_embedding 等原始数据, 以及利用feedid计算的embedding。
