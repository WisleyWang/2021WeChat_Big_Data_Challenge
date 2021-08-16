from torch import nn
import pandas as pd
import numpy as np
import pickle
import torchtext
from torch.optim import Optimizer
import torch
from deepctr_torch.models.deepfm import FM,DNN
from deepctr_torch.layers  import CIN,InteractingLayer,CrossNet,CrossNetMix,AttentionSequencePoolingLayer
from deepctr_torch.models.basemodel import *
from tqdm import tqdm
import random
import gc
from collections import defaultdict
from sklearn.metrics import auc,roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

import math
def disable_grad(module):
    for param in module.parameters():
        param.requires_grad=False
class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()
        self.att_emb=Attn(hidden_dims)
        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos), input_dims,
            padding_idx=field.vocab.stoi[field.pad_token])
        self.emb.weight.data.copy_(torch.from_numpy(field.vocab.vectors).float())
        self.emb.weight.requires_grad = False
    def forward(self, x):
        """
        x: (batch_size, max_length) LongTensor
        length: (batch_size,) LongTensor
        """
#         x = self.emb(x).sum(1)# / length.unsqueeze(1).float() # 归一化
        return  self.att_emb(self.emb(x))#self.proj(x)

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

class Transmodel(nn.Module):
    def __init__(self,user_data,feed_data,textset,feed_embed,graph_emb,device):
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
        self.att_pool1=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)
        self.att_pool2=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)
        self.att_pool3=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)

        self.mmoe=MMOELayer(sum(user_dict.values())+128*12, mmoe_hidden_dim=128,num_task=7,n_expert=5,expert_activation=None,device=device)
        
        self.liner1=nn.Linear(128,1)
        self.liner2=nn.Linear(128,1)
        self.liner3=nn.Linear(128,1)
        self.liner4=nn.Linear(128,1)
        self.liner5=nn.Linear(128,1)
        self.liner6=nn.Linear(128,1)
        self.liner7=nn.Linear(128,1)
    def forward(self,userid,feedid,hist,mask_leng,is_train=True):
        # hist=[B,T]  #T是padding的序列
        # mask_leng=[B,1] # 每个batch中的长度
        user_projections=[]
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
                text_embedding.append(result)
        user_feat=torch.cat(user_projections,-1)
        spare_emb=self.spare_liner(torch.cat(sparse_embedding,-1))
        dense_emb=self.dense_liner(torch.cat(dense_embedding,-1))
        text_emb=self.text_liner(torch.cat(text_embedding+[self.feed_embed,self.graph],-1))  
        feed_feat=torch.cat([spare_emb,dense_emb,text_emb],-1) #128*3
        
        hist_feat=feed_feat[hist]
        query=torch.unsqueeze(feed_feat[feedid],1)
#         print(query,hist_feat.shape,mask_leng.shape)
        
        att_output1=self.att_pool1(query,hist_feat,mask_leng)
        att_output1=att_output1.squeeze()
        att_output2=self.att_pool2(query,hist_feat,mask_leng)
        att_output2=att_output2.squeeze()
        att_output3=self.att_pool3(query,hist_feat,mask_leng)
        att_output3=att_output3.squeeze()
        combine=torch.cat([user_feat[userid],feed_feat[feedid],att_output1,att_output2,att_output3],axis=-1)
        outs=self.mmoe(combine)

        logit_gnn1=self.liner1(outs[0])#+ffm1#128+1+128*2
        logit_gnn2=self.liner2(outs[1])
        
        logit_gnn3=self.liner3(outs[2])
        logit_gnn4=self.liner4(outs[3])
        logit_gnn5=self.liner5(outs[4])
        logit_gnn6=self.liner6(outs[5])
        logit_gnn7=self.liner7(outs[6])

        return logit_gnn1,logit_gnn2,logit_gnn3,logit_gnn4,logit_gnn5,logit_gnn6,logit_gnn7
    
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
    def __init__(self, hidden_size,device, mmoe_hidden_dim=128,num_task=4,n_expert=3,expert_activation=None,):
        super(MMOELayer, self).__init__()
         # experts
        self.num_task=num_task
        self.expert_activation = expert_activation
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert).to(device), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert).to(device), requires_grad=True)
        # gates
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True).to(device) for _ in range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True).to(device) for _ in range(num_task)]
        for i in range(num_task):
            setattr(self, 'task_{}_dnn'.format(i+1),DNN(mmoe_hidden_dim,(128,128),dropout_rate=0.2,l2_reg=5e-5,use_bn=True))
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

def n_evaluate_nn(model,val_df,action_list,device,batch_size=512):
    model.eval()
    leng=len(val_df)
    val_src=val_df['userid'].apply(lambda x:userid2nid[x]).values
    val_dst=val_df['feedid'].apply(lambda x:feedid2nid[x]).values
    val_hist_id=torch.from_numpy((val_df['date_'].values-1)*len(userid2nid)+val_src).long()
    val_pred=[]
    all_aucs=[]
    weights=[0.30769231, 0.23076923, 0.15384615, 0.07692308, 0.07692308,0.07692308, 0.07692308]
    with torch.no_grad():
        for i in tqdm(range(0,leng//batch_size+1)):
            #         print(i*batch_size,(i+1)*batch_size)
            batch_src=val_src[i*batch_size:(i+1)*batch_size]
            batch_dst=val_dst[i*batch_size:(i+1)*batch_size]
            batch_hist=hist_seq[val_hist_id[i*batch_size:(i+1)*batch_size]]
            pred=model(batch_src,batch_dst,batch_hist[:,:-1],batch_hist[:,-1:].to(device))
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

def relative_position_encoding(depth, max_length=512, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(max_length)
    range_mat = range_vec.repeat(max_length).view(max_length, max_length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    embeddings_table = torch.zeros(vocab_size, depth)
    position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)

    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    positions_encoding = positions_encoding.view(my_shape)
    return positions_encoding
class NeZhaSelfAttention(nn.Module):
    def __init__(self, output_attentions,num_attention_heads,hidden_size,attention_probs_dropout_prob,max_relative_position,max_position_embeddings):
        super().__init__()
        self.output_attentions = output_attentions

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size /num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.relative_positions_encoding = relative_position_encoding(max_length=max_position_embeddings,
                                                                     depth=self.attention_head_size,
                                                                     max_relative_position=max_relative_position)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        batch_size, num_attention_heads, from_seq_length, to_seq_length = attention_scores.size()

        relations_keys = self.relative_positions_encoding[:to_seq_length, :to_seq_length, :].to(hidden_states.device)
        query_layer_t = query_layer.permute(2, 0, 1, 3)

        query_layer_r = query_layer_t.contiguous().view(from_seq_length, batch_size * num_attention_heads,
                                                        self.attention_head_size)
        key_position_scores = torch.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
        key_position_scores_r = key_position_scores.view(from_seq_length, batch_size,
                                                         num_attention_heads, from_seq_length)
        key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
        attention_scores = attention_scores + key_position_scores_r_t

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        relations_values = self.relative_positions_encoding[:to_seq_length, :to_seq_length, :].to(hidden_states.device)
        attention_probs_t = attention_probs.permute(2, 0, 1, 3)
        attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads,
                                                                 to_seq_length)
        value_position_scores = torch.matmul(attentions_probs_r, relations_values)
        value_position_scores_r = value_position_scores.view(from_seq_length, batch_size,
                                                             num_attention_heads, self.attention_head_size)
        value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
        context_layer = context_layer + value_position_scores_r_t

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs        
class Transmodelv2(nn.Module):
    def __init__(self,user_data,feed_data,textset,feed_embed,graph_emb,device):
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
        self.att_pool1=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)
        self.att_pool2=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)
        self.att_pool3=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)
        self.att_pool4=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)

        self.mmoe1=MMOELayer(sum(user_dict.values())+128*6, mmoe_hidden_dim=128,num_task=2,n_expert=5,expert_activation=None,device=device)
        self.mmoe2=MMOELayer(sum(user_dict.values())+128*6, mmoe_hidden_dim=128,num_task=2,n_expert=5,expert_activation=None,device=device)
        self.mmoe3=MMOELayer(sum(user_dict.values())+128*6, mmoe_hidden_dim=128,num_task=2,n_expert=5,expert_activation=None,device=device)
        self.mmoe4=MMOELayer(sum(user_dict.values())+128*6, mmoe_hidden_dim=128,num_task=1,n_expert=5,expert_activation=None,device=device)
      
        
        self.liner1=nn.Linear(128,1)
        self.liner2=nn.Linear(128,1)
        self.liner3=nn.Linear(128,1)
        self.liner4=nn.Linear(128,1)
        self.liner5=nn.Linear(128,1)
        self.liner6=nn.Linear(128,1)
        self.liner7=nn.Linear(128,1)
    def forward(self,userid,feedid,hist,mask_leng,is_train=True):
        # hist=[B,T]  #T是padding的序列
        # mask_leng=[B,1] # 每个batch中的长度
        user_projections=[]
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
                text_embedding.append(result)
        user_feat=torch.cat(user_projections,-1)
        spare_emb=self.spare_liner(torch.cat(sparse_embedding,-1))
        dense_emb=self.dense_liner(torch.cat(dense_embedding,-1))
        text_emb=self.text_liner(torch.cat(text_embedding+[self.feed_embed,self.graph],-1))  
        feed_feat=torch.cat([spare_emb,dense_emb,text_emb],-1) #128*3
        
        hist_feat=feed_feat[hist]
        query=torch.unsqueeze(feed_feat[feedid],1)
#        
        att_output1=self.att_pool1(query,hist_feat,mask_leng)
        att_output1=att_output1.squeeze()
        att_output2=self.att_pool2(query,hist_feat,mask_leng)
        att_output2=att_output2.squeeze()
        att_output3=self.att_pool3(query,hist_feat,mask_leng)
        att_output3=att_output3.squeeze()
        att_output4=self.att_pool4(query,hist_feat,mask_leng)
        att_output4=att_output4.squeeze()
        
        combine1=torch.cat([user_feat[userid],feed_feat[feedid],att_output1],axis=-1)
        combine2=torch.cat([user_feat[userid],feed_feat[feedid],att_output2],axis=-1)
        combine3=torch.cat([user_feat[userid],feed_feat[feedid],att_output3],axis=-1)
        combine4=torch.cat([user_feat[userid],feed_feat[feedid],att_output4],axis=-1)

        outs1=self.mmoe1(combine1)
        outs2=self.mmoe2(combine2)
        outs3=self.mmoe3(combine3)
        outs4=self.mmoe4(combine4)
 

        logit_gnn1=self.liner1(outs1[0])#+ffm1#128+1+128*2
        logit_gnn2=self.liner2(outs2[0])
        logit_gnn3=self.liner3(outs3[0])
        logit_gnn4=self.liner4(outs1[1])
        logit_gnn5=self.liner5(outs2[1])
        logit_gnn6=self.liner6(outs3[1])
        logit_gnn7=self.liner7(outs4[0])

        return logit_gnn1,logit_gnn2,logit_gnn3,logit_gnn4,logit_gnn5,logit_gnn6,logit_gnn7
    
class Transmodelv3(nn.Module):
    def __init__(self,user_data,feed_data,textset,feed_embed,graph_emb,device):
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
        self.att_pool1=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
                                                supports_masking=False)
#         self.att_pool2=AttentionSequencePoolingLayer(att_hidden_units=(128,128),embedding_dim=128*3, weight_normalization=True,
#                                                 supports_masking=False)
        
        self.satt=NeZhaSelfAttention(output_attentions=False,num_attention_heads=6,hidden_size=128*3,attention_probs_dropout_prob=0.2,\
                                           max_relative_position=128,max_position_embeddings=128)

        self.mmoe=MMOELayer(sum(user_dict.values())+128*6, mmoe_hidden_dim=128,num_task=7,n_expert=5,expert_activation=None,device=device)
        
        self.liner1=nn.Linear(128,1)
        self.liner2=nn.Linear(128,1)
        self.liner3=nn.Linear(128,1)
        self.liner4=nn.Linear(128,1)
#         self.liner5=nn.Linear(128,1)
        self.liner6=nn.Linear(128,1)
        self.liner7=nn.Linear(128,1)
    def forward(self,userid,feedid,hist,mask_leng,is_train=True):
        # hist=[B,T]  #T是padding的序列
        # mask_leng=[B,1] # 每个batch中的长度
        user_projections=[]
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
                text_embedding.append(result)
        user_feat=torch.cat(user_projections,-1)
        spare_emb=self.spare_liner(torch.cat(sparse_embedding,-1))
        dense_emb=self.dense_liner(torch.cat(dense_embedding,-1))
        text_emb=self.text_liner(torch.cat(text_embedding+[self.feed_embed,self.graph],-1))  
        feed_feat=torch.cat([spare_emb,dense_emb,text_emb],-1) #128*3
        
        hist_feat=feed_feat[hist]
        hist_feat = self.satt(hist_feat)[0]
        
        query=torch.unsqueeze(feed_feat[feedid],1)
#         print(query,hist_feat.shape,mask_leng.shape)
        
        att_output1=self.att_pool1(query,hist_feat,mask_leng)
        att_output1=att_output1.squeeze()
#         att_output2=self.att_pool2(query,hist_feat,mask_leng)
#         att_output2=att_output2.squeeze()
#         att_output3=self.att_pool3(query,hist_feat,mask_leng)
#         att_output3=att_output3.squeeze()
        combine=torch.cat([user_feat[userid],feed_feat[feedid],att_output1],axis=-1)
        outs=self.mmoe(combine)

        logit_gnn1=self.liner1(outs[0])#+ffm1#128+1+128*2
        logit_gnn2=self.liner2(outs[1])
        
        logit_gnn3=self.liner3(outs[2])
        logit_gnn4=self.liner4(outs[3])
        logit_gnn5=self.liner1(outs[4])
        logit_gnn6=self.liner6(outs[5])
        logit_gnn7=self.liner7(outs[6])

        return logit_gnn1,logit_gnn2,logit_gnn3,logit_gnn4,logit_gnn5,logit_gnn6,logit_gnn7
    
def test_pred_func(model,test_a,src,dst,hist_id,hist_seq,batch_size=4096*3):
    model.eval()
#     batch_size=4096
    leng=len(test_a)
    val_pred=[]
    with torch.no_grad():
        for i in tqdm(range(0,leng//batch_size+1)):
            #         print(i*batch_size,(i+1)*batch_size)
            batch_src=src[i*batch_size:(i+1)*batch_size]
            batch_dst=dst[i*batch_size:(i+1)*batch_size]
            batch_hist=hist_seq[hist_id[i*batch_size:(i+1)*batch_size]]
            pred=model(batch_src,batch_dst,batch_hist[:,:-1],batch_hist[:,-1:].cuda())
            val_pred.append(torch.cat(pred,axis=-1).sigmoid().cpu().numpy())
        val_pred=np.concatenate(val_pred,axis=0)
    return val_pred