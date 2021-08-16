#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
import gc
import os
import random
from tqdm import tqdm
from sklearn.metrics import auc,roc_auc_score
from deepctr_torch.models.deepfm import FM,DNN
from deepctr_torch.layers  import CIN,InteractingLayer,CrossNet,CrossNetMix
from deepctr_torch.models.basemodel import *
from collections import defaultdict
from torch.optim import Optimizer
import torchtext
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
class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos), input_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        self.emb.weight[:] = field.vocab.vectors
        self.proj = nn.Linear(input_dims, hidden_dims)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

        disable_grad(self.emb) # 词向量不可训练

    def forward(self, x):
        """
        x: (batch_size, max_length) LongTensor
        length: (batch_size,) LongTensor
        """
        x = self.emb(x).sum(1)# / length.unsqueeze(1).float() # 归一化
        return self.proj(x)

class BagOfWords(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()
        self.att_emb=Attn(hidden_dims)
        self.emb = nn.Embedding(
            len(field.vocab.itos), hidden_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x):
        return self.att_emb(self.emb(x))#.mean(1)#/ length.unsqueeze(1).float() # 归一化
class text_emb(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.att_emb=Attn(weight.shape[1])
        self.emb = nn.Embedding(
            weight.shape[0],weight.shape[1],
            padding_idx=0)
#         nn.init.xavier_uniform_(self.emb.weight)
        self.emb.weight.data.copy_(torch.from_numpy(weight).float())
#         self.emb.requires_grad_=False
    def forward(self, x):
        return self.att_emb(self.emb(x))#.mean(1)#/ length.unsqueeze(1).float() # 归一化
class Model1(nn.Module):
    def __init__(self,user_data,feed_data,textset,feed_embed,graph_emb):
        super().__init__()
        self.feed_data=feed_data
        self.user_data=user_data
        user_dict={'device':2,'userid':128}
        feed_dict={'bgm_song_id':16, 'bgm_singer_id':16,'authorid':16,'dense':32,'hash_dense':32
       ,'manual_keyword_id1':16,'manual_tag_id1':16,'machine_keyword_id1':16
            ,'machine_tag_id1':16,'knn_feed':16,
           'manual_tag_list':32,'manual_keyword_list':32,'machine_keyword_list':32,'asr':32,'description':32,'ocr':32
                  }
        self.model_dict=_init_input_modules(user_data,feed_data,textset, user_dict,feed_dict)
        self.spare_liner=nn.Linear(8*16,128)
        self.dense_liner=nn.Linear(32*2,128)
        self.text_liner=nn.Linear(32*6+512+64,128)
        self.feed_embed= nn.Parameter(torch.from_numpy(feed_embed).float(),requires_grad=False)
        self.graph= nn.Parameter(torch.from_numpy(graph_emb).float(),requires_grad=False)
        self.reg_liner=nn.Linear(128,1)
        self.dynami_dense=nn.Linear(96,64)
        self.cross1=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross2=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross3=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross4=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross5=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross6=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
        self.cross7=CrossNetMix(sum(user_dict.values())+128*3+64,layer_num=4)
#         self.dnn=DNN(sum(user_dict.values())+128*3+64,(128,128),dropout_rate=0.1)
        self.mmoe=MMOELayer(sum(user_dict.values())+128*3+64, mmoe_hidden_dim=128,num_task=6,n_expert=5,expert_activation=None)
#         self.att1=Attn(sum(user_dict.values())+128*3+64)
#         self.att2=Attn(sum(user_dict.values())+128*3+64)
#         self.att3=Attn(sum(user_dict.values())+128*3+64)
#         self.att4=Attn(sum(user_dict.values())+128*3+64)
        
        self.liner1=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
#         self.liner2=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
#         self.liner3=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
#         self.liner4=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
#         self.liner5=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
#         self.liner6=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
#         self.liner7=nn.Linear(128+sum(user_dict.values())+128*3+64,1)
    def forward(self,userid,feedid,batch_dense,is_train=True):
        user_projections=[]
#         feed_projections=[]
        dense_embedding=[]
        sparse_embedding=[]
        text_embedding=[]
        for feature, data in self.user_data.items():
            module = self.model_dict[feature]
            result = module(data)
            user_projections.append(result)
        for feature, data in self.feed_data.items():
#             print(feature)
            module = self.model_dict[feature]
            result = module(data)
            if result.shape[-1]==16:
                sparse_embedding.append(result)
            elif 'dense' in feature:
                dense_embedding.append(result)
            else:
#                 print(result.shape)
                text_embedding.append(result)
#         print(user_projections)

        user_feat=torch.cat(user_projections,-1)
        spare_emb=self.spare_liner(torch.cat(sparse_embedding,-1))
        dense_emb=self.dense_liner(torch.cat(dense_embedding,-1))
        text_emb=self.text_liner(torch.cat(text_embedding+[self.feed_embed,self.graph],-1))  
        feed_feat=torch.cat([spare_emb,dense_emb,text_emb],-1) #128*3
        dynami_dense=self.dynami_dense(batch_dense)
        combine=torch.cat([user_feat[userid],feed_feat[feedid],dynami_dense],axis=-1)
        cross1=self.cross1(combine)
        cross2=self.cross2(combine)
        cross3=self.cross3(combine)
        cross4=self.cross4(combine)
        cross5=self.cross5(combine)
        cross6=self.cross6(combine)
        cross7=self.cross7(combine)
        outs=self.mmoe(combine)

        
        logit_gnn1=self.liner1(torch.cat([outs[0],cross1],axis=-1))#+ffm1#128+1+128*2
        logit_gnn2=self.liner1(torch.cat([outs[1],cross2],axis=-1))#+ffm2
        
        logit_gnn3=self.liner1(torch.cat([outs[2],cross3],axis=-1))#+ffm3
        logit_gnn4=self.liner1(torch.cat([outs[3],cross4],axis=-1))#+ffm4
        logit_gnn5=self.liner1(torch.cat([outs[0],cross5],axis=-1))#+ffm3
        logit_gnn6=self.liner1(torch.cat([outs[2],cross6],axis=-1))#+ffm4
        logit_gnn7=self.liner1(torch.cat([outs[4],cross7],axis=-1))#+ffm4
        logit_reg=self.reg_liner(outs[5])
        return logit_gnn1,logit_gnn2,logit_gnn3,logit_gnn4,logit_gnn5,logit_gnn6,logit_gnn7,logit_reg
    
def _init_input_modules(user_data,feed_data,textset, user_dict,feed_dict):
    # We initialize the linear projections of each input feature ``x`` as
    # follows:
    # * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
    #   feature, and assume the range of ``x`` is 0..max(x).
    # * If ``x`` is a float one-dimensional feature, we assume that ``x`` is a
    #   numeric vector.
    # * If ``x`` is a field of a textset, we process it as bag of words.
    module_dict = nn.ModuleDict()
    for column, data in user_data.items():
#         if column in user_texts.keys():
#             continue
        if data.dtype == torch.float32: # 数值类型的特征
            assert data.ndim == 2
            m = nn.Linear(data.shape[1],user_dict[column]) # 数值特征 做个线性变换
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif data.dtype == torch.int64:
            assert data.ndim == 1  # 整形的单值特征做个embedding
            m = nn.Embedding(data.max() + 2, user_dict[column], padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
        module_dict[column] = m  # 不同的特征名字对应不同的处理moderl 这里或许可以加FM进去
    
    for column, data in feed_data.items():
        if column in textset.fields.keys():
            continue
        if column =='manuual_tag_list_emb':
            continue
        if data.dtype == torch.float32: # 数值类型的特征
            assert data.ndim == 2
            m = nn.Linear(data.shape[1],feed_dict[column]) # 数值特征 做个线性变换
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif data.dtype == torch.int64:
            assert data.ndim == 1  # 整形的单值特征做个embedding
            m = nn.Embedding(data.max() + 2, feed_dict[column], padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
        module_dict[column] = m  # 不同的特征名字对应不同的处理moderl 这里或许可以加FM进去
        
    if textset is not None:
        for column, field in textset.fields.items():
            if field.vocab.vectors:
                module_dict[column] = BagOfWordsPretrained(field,feed_dict[column])
            else:
                module_dict[column] = BagOfWords(field,feed_dict[column])
    return module_dict

class MMOELayer(nn.Module):
    def __init__(self, hidden_size, mmoe_hidden_dim=128,num_task=4,n_expert=3,expert_activation=None):
        super(MMOELayer, self).__init__()
         # experts
        self.num_task=num_task
        self.expert_activation = expert_activation
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert).cuda(), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert).cuda(), requires_grad=True)
        # gates
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True).cuda() for _ in range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True).cuda() for _ in range(num_task)]
        for i in range(num_task):
            setattr(self, 'task_{}_dnn'.format(i+1),DNN(mmoe_hidden_dim,(128,128),dropout_rate=0.4,l2_reg=5e-5,use_bn=True,seed=1090))
    def forward(self,x):
         # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', x, self.experts) # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)
        
        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum('ab, bc -> ac',x, gate) # batch * num_experts
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)
        
        
        
        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1) # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(experts_out) # batch * mmoe_hidden_size * num_experts
            outs.append(torch.sum(weighted_expert_output, 2)) # batch * mmoe_hidden_size
          # task tower
        task_outputs = list()
        for i in range(self.num_task):
            oo = outs[i]
            mod=getattr(self, 'task_{}_dnn'.format(i+1))
            oo = mod(oo)
            task_outputs.append(oo)
        
        return task_outputs

class HighwayMLP(nn.Module):
    def __init__(self,
                 input_size,
                 gate_bias=-3,
                 activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):

        super(HighwayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = DNN(input_size,(input_size,input_size,input_size),dropout_rate=0.1)

        self.gate_layer = nn.Linear(input_size,input_size)

        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)
class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

from torch.optim.lr_scheduler import LambdaLR
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
        Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
        Linearly decreases the multiplicative variable from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
class Attn(nn.Module):
    def __init__(self,hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size,1)
    def forward(self, x):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''   
        att=self.attn(x)
        att=F.tanh(att)
        att=F.softmax(att,1)
        att_x=att*x
        return att_x.sum(1)   
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)
        return loss
    
def build_optimizer(model, train_steps, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False, eps=1e-8)
    optimizer = Lookahead(optimizer, 5, 1)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * 0.1, t_total=train_steps)
    return optimizer, scheduler

def n_evaluate_nn(val_df,action_list,batch_size=512):
    model.eval()
    leng=len(val_df)
    val_src=val_df['userid'].apply(lambda x:userid2nid[x]).tolist()
    val_dst=val_df['feedid'].apply(lambda x:feedid2nid[x]).tolist()
    val_dense=torch.from_numpy(val_df[feat].values).float()
#     regs=torch.from_numpy(train_ratings['reg'].values).float()
    val_pred=[]
    all_aucs=[]
    weights=[0.30769231, 0.23076923, 0.15384615, 0.07692308, 0.07692308,0.07692308, 0.07692308]
    with torch.no_grad():
        for i in tqdm(range(0,leng//batch_size+1)):
            #         print(i*batch_size,(i+1)*batch_size)
            batch_src=val_src[i*batch_size:(i+1)*batch_size]
            batch_dst=val_dst[i*batch_size:(i+1)*batch_size]
            batch_dense=val_dense[i*batch_size:(i+1)*batch_size].cuda()
            pred=model(batch_src,batch_dst,batch_dense)
            val_pred.append(torch.cat(pred,axis=-1).sigmoid().cpu().numpy())
        val_pred=np.concatenate(val_pred,axis=0)
        for i,action in enumerate(action_list):
            val_df['pred_'+action]=val_pred[:,i]
            label_nunique = val_df.groupby(by='userid')[action].transform('nunique')
            tmp_df = val_df[label_nunique == 2]
            aucs = tmp_df.groupby(by='userid').apply(
                lambda x: roc_auc_score(x[action].values, x['pred_'+action].values))
            all_aucs.append(np.mean(aucs))
            print('val %s uauc:'%action,np.mean(aucs))
            print('val %s auc:'%action,roc_auc_score(val_df[action].values,val_pred[:,i]))
        print('score uauc:',sum([all_aucs[i]*weights[i] for i in range(len(action_list))]))
        

def test_pred_func(model,test_df,test_src,test_dst,test_dense,batch_size): 
#     test_src=test_df['userid'].apply(lambda x:userid2nid[x]).tolist()
#     test_dst=test_df['feedid'].apply(lambda x:feedid2nid[x]).tolist()
#     batch_size=4096*2
#     test_dense=torch.from_numpy(test_df[feat].values).float()
    test_pred=[]
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0,len(test_df)//batch_size+1)):
    #         print(i*batch_size,(i+1)*batch_size)
            batch_src=test_src[i*batch_size:(i+1)*batch_size]
            batch_dst=test_dst[i*batch_size:(i+1)*batch_size]
            batch_dense=test_dense[i*batch_size:(i+1)*batch_size].cuda()
            pred=model(batch_src,batch_dst,batch_dense)
            pred=torch.cat(pred,axis=-1)
            test_pred.append(pred.sigmoid().cpu().numpy())
    test_pred=np.concatenate(test_pred,axis=0)
    return test_pred
