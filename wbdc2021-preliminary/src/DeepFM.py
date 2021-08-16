# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA

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
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_b.csv"
# dict
DICT_KEYWORD = FEATURE_PATH + "dict_keyword.npy"
DICT_OCR = FEATURE_PATH + "dict_ocr.npy"
DICT_TAG = FEATURE_PATH + "dict_tag.npy"

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'description', 'ocr', 'asr',
                'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list',
                'machine_tag_list']

embeddings = ['embed_0',
       'embed_1', 'embed_2', 'embed_3', 'embed_4', 'embed_5', 'embed_6',
       'embed_7', 'embed_8', 'embed_9', 'embed_10', 'embed_11', 'embed_12',
       'embed_13', 'embed_14', 'embed_15', 'embed_16', 'embed_17', 'embed_18',
       'embed_19', 'embed_20', 'embed_21', 'embed_22', 'embed_23', 'embed_24',
       'embed_25', 'embed_26', 'embed_27', 'embed_28', 'embed_29', 'embed_30',
       'embed_31']

graph_embedding = ['0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
       '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
       '30', '31']

tag_embedding = ['tag_0','tag_1','tag_2','tag_3','tag_4','tag_5','tag_6','tag_7',
                                                         'tag_8','tag_9','tag_10','tag_11','tag_12','tag_13','tag_14',
                                                         'tag_15','tag_16','tag_17','tag_18','tag_19','tag_20','tag_21',
                                                         'tag_22','tag_23','tag_24','tag_25','tag_26','tag_27','tag_28',
                                                         'tag_29','tag_30','tag_31']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}

NUM_EPOCH_DICT = {"read_comment": 2, "like": 2, "click_avatar": 2,"forward": 2,
                                "comment": 1, "follow": 1, "favorite": 1, }

class MyBaseModel(BaseModel):

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                try:
                                    temp = metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                                except Exception:
                                    temp = 0
                                finally:
                                    train_result[name].append(temp)
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                temp = metric_fun(y, pred_ans)
            except Exception:
                temp = 0
            finally:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

class MyDeepFM(MyBaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0.0001, init_std=0.0001, seed=1024,
                 dnn_dropout=0.5,
                 dnn_activation='relu', dnn_use_bn=True, task='binary', device='cpu', gpus=None):

        super(MyDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        #rint('x',X.shape,X)

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)
        #print(logit.shape,logit)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)
            
        #print('fm',logit.shape,logit)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit
        #print('logit',logit.shape,logit)

        y_pred = self.out(logit)
        #print(y_pred.shape,y_pred)

        return y_pred


if __name__ == "__main__":
    for idx,seed in enumerate([0,38,57,150,2021,6666]):
        submit = pd.read_csv(DATASET_PATH + 'test_b.csv')[['userid', 'feedid']]
        for action in ACTION_LIST:
            USE_FEAT = ['userid', 'feedid', action] + FEA_FEED_LIST[1:] + embeddings + graph_embedding
            train = pd.read_csv(FEATURE_PATH + f'/train_data_for_{action}.csv')[USE_FEAT]
            train = train.sample(frac=1, random_state=42).reset_index(drop=True)
            #train = train[train['play']!=0]
            print("posi prop:")
            
            #print(train.columns)
            print(sum((train[action]==1)*1)/train.shape[0])
            test = pd.read_csv(FEATURE_PATH + '/test_data.csv')[[i for i in USE_FEAT if i != action]]
            #test = test[test['play']!=0]
            target = [action]
            test[target[0]] = 0
            test = test[USE_FEAT]
            data = pd.concat((train, test)).reset_index(drop=True)
            
            ## load dict
            dict_keyword = np.load(DICT_KEYWORD, allow_pickle=True).item()
            dict_ocr = np.load(DICT_OCR, allow_pickle=True).item()
            dict_tag = np.load(DICT_TAG, allow_pickle=True).item()
            #print(len(dict_keyword))
            key2index = {'description':dict_ocr, 'ocr':dict_ocr, 'asr':dict_ocr, 'manual_keyword_list':dict_keyword, 'machine_keyword_list':dict_keyword, 'manual_tag_list':dict_tag}
            maxlen = {'description':200, 'ocr':200, 'asr':200, 'manual_keyword_list':5, 'machine_keyword_list':5, 'manual_tag_list':5}
            
            dense_features = ['videoplayseconds']
            varlen_features = ['description', 'ocr', 'asr', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list']
            sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'machine_tag_list']
            embedding = ['embedding', 'graph_embedding']
            

            data[sparse_features] = data[sparse_features].fillna(0)
            data[dense_features] = data[dense_features].fillna(0)
            
            print('*********************************************')
            print('sparse_features:',sparse_features)
            print('dense_features:',dense_features)
            print('varlen_features:',varlen_features)
            print('************************************')
            

            # 1.Label Encoding for sparse features,and do simple Transformation for dense features
            for feat in sparse_features:
                lbe = LabelEncoder()
                data[feat] = lbe.fit_transform(data[feat])
            mms = MinMaxScaler(feature_range=(0, 1))
            data[dense_features] = mms.fit_transform(data[dense_features])

            feature_names = dense_features + varlen_features + sparse_features
            print('feature_names',feature_names)

            # 3.generate input data for model
            ## fixlen_features
            train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(drop=True)
            train_model_input = {name: train[name] for name in feature_names}
            test_model_input = {name: test[name] for name in feature_names}
            
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),embedding_dim=8)
                                    for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features] +[DenseFeat(feat, 32, ) for feat in embedding]
            
            ##varlen_features
            varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat,vocabulary_size=len(key2index[feat]) + 1, embedding_dim=8), maxlen=maxlen[feat], combiner='mean') for feat in varlen_features]

            dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
            linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
            #print('train_model_input:',train_model_input)

            # 4.Define Model,train,predict and evaluate
            device = 'cpu'
            use_cuda = True
            if use_cuda and torch.cuda.is_available():
                print('cuda ready...')
                device = 'cuda:1'
            
            # 5.k-fold
            folds = StratifiedKFold(n_splits=5,random_state=seed,shuffle=True)
            predictions = np.zeros([len(test),1])
            print('start predicting ', action)
            print(predictions.shape,folds.n_splits)
            
            mms = mms.fit(train[embeddings])
            '''
            test_embeddings = np.array(test[embeddings])
            test_embeddings[np.isnan(test_embeddings)] = 0
            
            test_model_input['embedding'] = np.array(test_embeddings)
            
            test_graph_embeddings = np.array(test[graph_embedding])
            test_graph_embeddings[np.isnan(test_graph_embeddings)] = 0
            
            test_model_input['graph_embedding'] = np.array(test_graph_embeddings)
            '''
            
            for feat in ['embedding','graph_embedding']:
                print('add feat',feat)
                if feat == 'embedding': 
                    embedding = embeddings
                elif feat == 'graph_embedding': 
                    embedding = graph_embedding
                else:
                    embedding = tag_embedding
                test_embeddings = np.array(test[embedding])
                test_embeddings[np.isnan(test_embeddings)] = 0

                test_model_input[feat] = np.array(test_embeddings)
            
            def split(x,feat):
                key = []
                if feat in ['description','ocr','asr']:
                    key_ans = x.split(' ')
                else:
                    key_ans = x.split(';')
                return list(map(lambda x: key2index[feat][x], key_ans))
            
            dict_train = {}
            dict_test = {}
            for feat in varlen_features:
                train[feat].fillna('-1', inplace = True)
                    
                print('start calculating VarLenSparseFeat',feat)

                genres_list_train = list(map(split, train[feat].values,[feat]*len(train)))
                # Notice : padding=`post`
                genres_list_train = pad_sequences(genres_list_train, maxlen=maxlen[feat], padding='post', )
                dict_train[feat] = np.array(genres_list_train)

                test[feat].fillna('-1', inplace = True)
                genres_list_test = list(map(split, test[feat].values,[feat]*len(test)))
                # Notice : padding=`post`
                genres_list_test = pad_sequences(genres_list_test, maxlen=maxlen[feat], padding='post', )
                dict_test[feat] = np.array(genres_list_test)

            
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[feature_names].values, train[target].values)):
                print("Fold {}".format(fold_))

                train_model_input = {name: train.iloc[trn_idx][name] for name in feature_names}
                val_model_input = {name: train.iloc[val_idx][name] for name in feature_names}
                
                for feat in ['embedding','graph_embedding']:
                    print('add feat',feat)
                    if feat == 'embedding': 
                        embedding = embeddings
                    elif feat == 'graph_embedding': 
                        embedding = graph_embedding
                    else:
                        embedding = tag_embedding
                    train_embeddings = np.array(train.iloc[trn_idx][embedding])
                    train_embeddings[np.isnan(train_embeddings)] = 0

                    val_embeddings = np.array(train.iloc[val_idx][embedding])
                    val_embeddings[np.isnan(val_embeddings)] = 0

                    train_model_input[feat] = np.array(train_embeddings)
                    val_model_input[feat] = np.array(val_embeddings)
                
                
                for feat in varlen_features:
                    print('add feat',feat)
                    train_model_input[feat] = dict_train[feat][trn_idx]
                    val_model_input[feat] = dict_train[feat][val_idx]
                    test_model_input[feat] = dict_test[feat]
                
                model = MyDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                            task='binary',
                            l2_reg_embedding=1e-1, device=device)

                model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
                #print(train_model_input)

                history = model.fit(train_model_input, train.iloc[trn_idx][target].values, batch_size=1024, epochs=NUM_EPOCH_DICT[action],                                   verbose=1,validation_split=0,validation_data=(val_model_input, train.iloc[val_idx][target].values))

                pred_ans = model.predict(test_model_input, 128)
                predictions += pred_ans/folds.n_splits

                submit[action] = predictions
                torch.cuda.empty_cache()
        # 保存提交文件
        torch.save(model.state_dict(), MODEL_PATH+'DeepFM_b_weights_{}.h5'.format(idx))
        submit.to_csv(SUBMIT_PATH+"/submit_b_deepfm_{}.csv".format(idx), index=False)
