# 环境依赖
Python 3.6.5
所需安装包在requirement.txt中,推断前,需要source 打包好的anaconda虚拟环境,虚拟环境名称为tf1:
```
source active envs/tf1/
```

# 目录结构
```
├── data
│   ├── model (存放模型.pth)
│   ├── submission (存放提交结果)
│   ├── tmp (存放特征)
│   └── wedata(数据)
│       ├── wechat_algo_data1
│       └── wechat_algo_data2
└── src
    ├── prepare(提取特征代码)
    └── train (训练代码)
        └── model(模型的backbone)
```

# 运行流程
- bash  #进入bash
- source active envs/tf1/   # 激活虚拟环境
- bash inference.sh xxx/xx/xx.csv  # 运行推断脚本,输入参数为测试集路径

# 模型及特征
进行了多个模型的融合,主题框架是MMOE+crossNet ,训练的方式是借鉴了GNN的方式,先构建userid 和feedid 的异构图表,build模型的时候直接作为模型参数\
训练时只需要输入对应的userid 和feedid,取相应特征表在模型内部进行拼接,可以大大减少内存和显存的消耗,初赛只用了7G左右的显存,6G的内存.

统计特征使用了taeget encoding ,在初赛的时候效果显著,但是复赛发现作用不大,主要起作用的是原始的ID,文本等特征直接做embedding.
**值得一提**的是,对feedid按照userid进行groupby,然后进行w2v的预训练,效果明显,这个类似于GNN中对feed进行消息传递的作用,也对比了randomwalk以及n2v,效果都差不多.

# 算法性能【重要】
在复赛【A榜】测试集上,单模型从对测试集特征处理,到加载模型,模型推断,总耗时大概2分钟到3分钟,每2000条的耗时在7ms以内.

inference.sh 
总预测时长: 773 s
单个目标行为2000条样本的平均预测时长: 51.9408 ms

# 代码说明【重要】

| 路径             | 行数 | 内容                                |
| ---------------- | ---- | ----------------------------------- |
| src/inference.py | 80  | test_pred=test_pred_func(model,test_a,src,dst,hist_id,hist_seq,batch_size=4096*3) |
| src/inference.py | 86  | test_pred=test_pred_func(model,test_a,src,dst,hist_id,hist_seq,batch_size=4096*3) |
| src/inference.py | 92  |test_pred=test_pred_func(model,test_a,src,dst,batch_size=4096*3) |
| src/inference.py | 102 | test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3) |
| src/inference.py | 111 |test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3) |
| src/inference.py |120 | test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3) |
| src/inference.py |120 | test_pred=test_pred_func(model,test_a,src,dst,test_dense,batch_size=4096*3) |
| src/inference.py |128,129 | sub[PREDICT_LIST]=(subt1[PREDICT_LIST]*0.5+subt2[PREDICT_LIST]*0.25+subt3[PREDICT_LIST]*0.25)/2+(sub1[PREDICT_LIST]*0.3+sub2[PREDICT_LIST]*0.2+sub3[PREDICT_LIST]*0.2+sub4[PREDICT_LIST]*0.2+sub1_1[PREDICT_LIST]*0.1)/2|
       
          
最终结果文件输出到 ./data/submission 下
```
sub.to_csv(os.path.join(ROOT_PATH+'submission','result.csv'),index=False)
```
