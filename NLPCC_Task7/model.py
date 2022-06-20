import torch
import torch.nn as nn
from gcn_net import GCN
import math
from cnn import *
import torch.nn.functional as F

class LastHiddenModel(nn.Module):
    """
    60
    """

    def __init__(self, model, dropout):

        super().__init__()


        self.bert = model
        self.out1 = nn.Linear(self.bert.config.hidden_size , 4)
        self.out2 = nn.Linear(self.bert.config.hidden_size , 4)
        self.out3 = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, data):
        input_ids, seg_ids, attention_mask, input_ids_a, seg_ids_a, attention_masks_a, label, _ = data
        last_hidden_state = self.bert(input_ids=input_ids, \
                             attention_mask=seg_ids, token_type_ids=attention_mask).last_hidden_state


        # last_hidden_state = outputs[0] # 所有字符最后一层hidden state # 32 400 768 ，但是PAD PAD

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask

        logits1 = self.out1(mean_embeddings)
        logits2 = self.out2(mean_embeddings)
        logits3 = self.out3(mean_embeddings)

        return logits1,logits2,logits3

class LastFourModel(nn.Module):
    """
    0.55
    """

    def __init__(self,model,dropout):
        super().__init__()

        self.bert = model
        self.bert.config.output_hidden_states =True
        self.out1 = nn.Linear(self.bert.config.hidden_size * 4, 4)
        self.out2 = nn.Linear(self.bert.config.hidden_size * 4, 4)
        self.out3 = nn.Linear(self.bert.config.hidden_size *4, 3)



    def forward(self, data):
        input_ids, seg_ids, attention_masks, input_ids_a, seg_ids_a, attention_masks_a, label, _ = data
        all_hidden_states = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks).hidden_states


        concatenate_pooling = torch.cat(

            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1

        )

        concatenate_pooling = concatenate_pooling[:, 0]

        output1 = self.out1(concatenate_pooling)
        output2 = self.out2(concatenate_pooling)
        output3 = self.out3(concatenate_pooling)

        return output1,output2,output3


class Bisa_Model_joined_task3(nn.Module):
    def __init__(self, model, dropout):
        super(Bisa_Model_joined_task3, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.bert = model
        self.out1 = nn.Linear(self.bert.config.hidden_size , 4)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 4)
        self.out3 = nn.Linear(self.bert.config.hidden_size, 3)

    def forw(self, data):
        input_ids, seg_ids, attention_masks, input_ids_a, seg_ids_a, attention_masks_a, label,_ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state

        query = last_hidden[:, 0]

        out1 = self.out1(query)
        out2 = self.out2(query)
        out3 = self.out3(query)

        return out1,out2,out3

class Bisa_Model_joined_cnn_task3(nn.Module):
    def __init__(self, model, dropout):
        super(Bisa_Model_joined_cnn_task3, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.cnn = IDCNN()
        self.bert = model
        self.out1 = nn.Linear(self.bert.config.hidden_size , 4)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 4)
        self.out3 = nn.Linear(self.bert.config.hidden_size, 3)


    def forward(self, data):
        input_ids, seg_ids, attention_masks, input_ids_a, seg_ids_a, attention_masks_a, label,_ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state

        query = last_hidden[:, 0]
        query = self.cnn(query)
        out1 = self.out1(query)
        out2 = self.out2(query)
        out3 = self.out3(query)

        return out1,out2,out3

class Bisa_Model2(nn.Module):
    """
    上下文 + growp
    """

    def __init__(self, model, dropout):
        super(Bisa_Model2, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.bert = model
        self.out = nn.Linear(self.bert.config.hidden_size * 2, 4)

    def forward(self, data):
        input_ids, seg_ids, attention_masks, input_ids_a, seg_ids_a, attention_masks_a, label,_ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state

        last_hidden_a = self.bert(input_ids=input_ids_a, \
                                attention_mask=seg_ids_a, token_type_ids=attention_masks_a, \
                                ).last_hidden_state

        query = last_hidden[:, 0]
        query_a = last_hidden_a[:,0]

        context_vec = torch.cat((query, query_a), dim=1)

        out = self.out(context_vec)
        out = self.softmax(out)
        return out


def get_types_tensor(types,types_softmax):
    _, indices = torch.max(types_softmax, 1)

    types_tensor = torch.Tensor([]).to(indices.device)
    # types_cuda = types.to(indices.device).detach()
    types_cuda = types.to(indices.device)

    for i in indices:
        types_tensor = torch.cat([types_tensor,types_cuda[i,:]],dim=0)
    types_tensor = types_tensor.view(len(indices), -1)
    types_tensor = torch.unsqueeze(types_tensor,1)
    return types_tensor # [batch,768]


class Bisa_Model_joined_concat(nn.Module):
    """
    joined attention
    """

    def __init__(self, model, types, dropout):
        super(Bisa_Model_joined_concat, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.bert = model


        self.out1 = nn.Linear(self.bert.config.hidden_size, 4)
        self.out2 = nn.Linear(self.bert.config.hidden_size*2, 4)
        self.out3 = nn.Linear(self.bert.config.hidden_size, 4)

        self.m = 0.2
        self.types = types

    def forward(self, data):
        input_ids, seg_ids, attention_masks, input_ids_a, seg_ids_a, attention_masks_a, label, _ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state
        type_tensor = self.bert(input_ids=self.types[0].to(last_hidden.device), \
                                attention_mask=self.types[1].to(last_hidden.device), token_type_ids=self.types[2].to(last_hidden.device), \
                                ).last_hidden_state[:, 0]

        # query = last_hidden[:, 0].detach()
        query = last_hidden[:, 0]
        out2 = self.out1(query)  # type pred

        type_tensor = get_types_tensor(type_tensor, torch.softmax(out2, dim=-1))
        type_tensor = torch.squeeze(type_tensor, dim=-2)
        out_cat = self.out2(torch.cat([query,type_tensor],dim=-1))

        cls_out = self.out3(query)

        out_cat = F.softmax(out_cat)
        cls_out = F.softmax(cls_out)

        out1 = out_cat * self.m + cls_out * (1 - self.m)


        return out1, out2

class Bisa_Model_joined_att(nn.Module):
    """
    joined attention
    """

    def __init__(self, model,types, dropout):
        super(Bisa_Model_joined_att, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.bert = model

        self.linear1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out3 = nn.Linear(self.bert.config.hidden_size, 4)

        self.out1 = nn.Linear(self.bert.config.hidden_size , 4)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 4)

        self.m = 0
        self.types = types

    def forward(self, data):
        input_ids, seg_ids, attention_masks, input_ids_a, seg_ids_a, attention_masks_a, label,_ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state

        # query = last_hidden[:, 0].detach()
        query = last_hidden[:, 0]
        out2 = self.out1(query)  # type pred

        type_tensor = get_types_tensor(self.types,torch.softmax(out2,dim=-1))

        # attention
        # last_hidden_linear = self.linear1(last_hidden)
        # # [batch,seq_len,hidden_size] * [batch,1,hidden_size] -> [batch,seq_len,1]
        # att_w = torch.matmul(last_hidden_linear,type_tensor.transpose(1,2))
        # att_w = torch.softmax(att_w,dim=1)
        # # [batch,seq_len,hidden_size] . [batch,seq_len,1] -> [batch,seq_len,hidden_size]
        # last_hidden_type = torch.mul(last_hidden,att_w)
        # last_hidden_type,_ = torch.max(last_hidden_type,dim=1)


        # attention cls
        type_tensor = torch.squeeze(type_tensor,dim=-2)
        att_w = F.softmax(torch.mul(query,type_tensor),dim=-1)
        cls_type = torch.mul(query,att_w)


        att_out = self.out2(cls_type)
        cls_out = self.out3(query)



        att_out = F.softmax(att_out)
        cls_out = F.softmax(cls_out)

        out1 = att_out * self.m + cls_out * (1 - self.m)

        return out1,out2

class Bisa_Model_joined(nn.Module):
    """
    joined
    """

    def __init__(self, model, dropout):
        super(Bisa_Model_joined, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.bert = model
        self.out1 = nn.Linear(self.bert.config.hidden_size , 4)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, data):
        input_ids, seg_ids, attention_masks, input_ids_a, seg_ids_a, attention_masks_a, label,_ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state

        query = last_hidden[:, 0]
        out1 = self.relu(self.out1(query))

        out2 = self.relu(self.out2(query))
        return out1,out2


class Bisa_Model_Gcn(nn.Module):
    """
    上下文
    """

    def __init__(self, model, dropout):
        super(Bisa_Model_Gcn, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.gcn = GCN(768,768,4,dropout)
        self.m = 0.7

        self.bert = model
        self.out = nn.Linear(self.bert.config.hidden_size, 4)


    def forward(self, data):
        input_ids, seg_ids, attention_masks, graph, _, _, _,_ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state

        query_bert = last_hidden[:, 0]
        cls_pred = self.out(query_bert)
        cls_pred = nn.Softmax(dim=1)(cls_pred)

        query_gcn = self.gcn(last_hidden,graph)
        gcn_out = torch.sum(query_gcn,dim=1)
        gcn_pred = nn.Softmax(dim=1)(gcn_out)


        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred

class Bisa_Model(nn.Module):
    """
    上下文
    """

    def __init__(self, model, dropout):
        super(Bisa_Model, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, data):
        input_ids, seg_ids, attention_masks, _, _, _, _,_ = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state[:, 0]

        out = self.out(last_hidden)

        out = self.relu(out)
        return out

class Bisa_Model_attentin(nn.Module):
    """
    上下文
    """

    def __init__(self, model, dropout):
        super(Bisa_Model_attentin, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.attention = SelfAttention(768,8,0.2)

        self.bert = model
        for param in self.bert.parameters():
            param.requires_grad = True

        self.out = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, data):
        input_ids, seg_ids, attention_masks, _, _, _, _,_,matrix = data
        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state

        query = self.attention(last_hidden,attention_masks)[:,0]
        out = self.out(query)
        out = self.relu(out)
        return out


class Bisa_Model_Lstm(nn.Module):
    def __init__(self, model, drop_prob=0.5):
        super(Bisa_Model_Lstm, self).__init__()

        self.n_layers = 2
        self.hidden_dim = 384
        self.bidirectional = True

        self.bert = model
        self.lstm = nn.LSTM(768, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional)

        self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, 4)
        self.relu = nn.ReLU()

    def forward(self, data):
        input_ids, seg_ids, attention_masks, _, _, _, _, _ = data

        last_hidden = self.bert(input_ids=input_ids, \
                                attention_mask=seg_ids, token_type_ids=attention_masks, \
                                ).last_hidden_state[:,0]
        #
        # lstm_out, (hidden_last, cn_last) = self.lstm(last_hidden)
        # # 正向最后一层，最后一个时刻
        # hidden_last_L = hidden_last[-2]
        # # 反向最后一层，最后一个时刻
        # hidden_last_R = hidden_last[-1]
        # # 进行拼接
        # hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        out = self.linear(last_hidden)

        out = self.out(out)
        out = self.relu(out)
        return out



class BertPooler(nn.Module):
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):

        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads  # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)  # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

    def forward(self, hidden_states, attention_mask):
        # eg: attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])  shape=[bs, seqlen]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, seqlen] 增加维度
        attention_mask = (1.0 - attention_mask) * -10000.0  # padding的token置为-10000，exp(-1w)=0

        # 线性变换
        mixed_query_layer = self.query(hidden_states)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)  # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores + attention_mask
        # 加上mask，将padding所在的表示直接-10000

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer  # [bs, seqlen, 128] 得到输出

from focal_loss import FocalLoss
class Metamodel(nn.Module):

    def __init__(self,model,dropout):
        self.model = Bisa_Model(model,dropout)
        self.loss = FocalLoss(num_class=4)

    def forward(self, data):
        return self.model(data)

    def loss_funtion(self,pre1,label1,pre2,label2):
        loss1 = self.loss(pre1,label1)
        loss2 = self.loss(pre2,label2)
        return



    def local_update(self, support_data, support_y):
        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None
        support_set_y_pred = self.model(support_data)

        loss = self.loss(support_y[:, 0].float(), support_set_y_pred[0], support_y[:, 1].float(), support_set_y_pred[1], device='cuda')

        self.model.zero_grad()
        grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
        fast_parameters = []
        for k, weight in enumerate(self.model.parameters()):
            if grad[k] is None:
                continue
            # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
            if weight.fast is None:
                weight.fast = weight - self.local_lr * grad[k]  # create weight.fast
            else:
                weight.fast = weight.fast - self.local_lr * grad[k]
            fast_parameters.append(weight.fast)

        return loss

    def global_update(self, data1, data2):
        batch_sz = data1[0].shape[0]
        losses_q = []
        for i in range(batch_sz):
            loss_sup = self.local_update(data1[0][i], data1[1][i],data1[2][i],data1[3][i])
            query_set_y_pred = self.model(data2[0])
            loss = self.loss_funtion(list_qry_y[i][:, 0].float(), query_set_y_pred[0])

            losses_q.append(loss)
        losses_q = torch.stack(losses_q).mean(0)
        return losses_q