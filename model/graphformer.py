"""
GCN model for relation extraction.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import Transformer
from torch.autograd import Variable
import numpy as np
from model.graph import head_to_graph, tree_to_adj
from utils import constant, torch_utils


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)

        # print("##################the num_class of Classifier is {}".format(logits.shape[-1]) )
        return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = DCGCN(opt, embeddings)

        # mlp output layer
        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning

        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, deprel, head, first_pos, second_pos = inputs  # unpack

        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, l):
            trees = head_to_graph(head, l)
            adj = [tree_to_adj(maxlen, trees[i], directed=False, self_loop=False).reshape(1, maxlen, maxlen) for i
                   in range(len(trees))]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            adj = Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)
            return adj

        adj = inputs_to_tree_reps(head.data, l)
        h, pool_mask = self.gcn(adj, inputs)

        # pooling
        first_mask, second_mask = first_pos.eq(0).eq(0).unsqueeze(2), second_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        first_out = pool(h, first_mask, type=pool_type)
        second_out = pool(h, second_mask, type=pool_type)

        outputs = torch.cat([h_out, first_out, second_out], dim=1)
        outputs = self.out_mlp(outputs)

        return outputs, h_out


class DCGCN(nn.Module):
    def __init__(self, opt, embeddings):
        super().__init__()
        self.opt = opt
        self.in_dim = opt['emb_dim'] + opt['pos_dim']
        self.emb, self.pos_emb = embeddings
        self.use_cuda = opt['cuda']
        self.mem_dim = opt['hidden_dim']

        # rnn layer
        self.input_W_R = nn.Linear(self.in_dim, self.mem_dim)
        # self.input_W_R2 = nn.Linear(600, 300)

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.num_layers = opt['num_layers']

        # self.layers = nn.ModuleList()

        self.heads = opt['heads']

        # 以下是Graph+Transformer模块的迭代，其中每次迭代都会包含多层的GCN和一层的Transformer
        self.block_num = opt['block_num']
        self.graphformer_layers = nn.ModuleList()
        self.aggregator = opt['aggregator']
        self.module_sequence_selector = opt['module_sequence_selector']

        for gf_i in range(self.block_num):
            layers = nn.ModuleList()
            # ---------------------------------------------------------
            # 该部分内容为每个GraphFormer block中，transformer放在gcn的前面
            if self.module_sequence_selector == 0:
                layers.append(Transformer(self.mem_dim))
            # ---------------------------------------------------------
            # 构建多层gcn的模块
            for i in range(self.num_layers):
                if i == 0:
                    #self.layers.append(GraphConvLayer(opt, self.mem_dim, 2))
                    layers.append(GraphConvLayer(opt, self.mem_dim, 4))
                    #self.layers.append(GraphConvLayer(opt, self.mem_dim, 5))
                else:
                    #self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, 2, self.heads))
                    layers.append(MultiHeadAttention(self.heads, self.mem_dim))
                    layers.append(MultiGraphConvLayer(opt, self.mem_dim, 4, self.heads))
                    #self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, 5, self.heads))
            # ---------------------------------------------------------
            # 该部分内容为每个GraphFormer block中，transformer放在gcn的后面
            if self.module_sequence_selector == 1:
                layers.append(Transformer(self.mem_dim))
            # ---------------------------------------------------------
            self.graphformer_layers.append(layers)

        # 聚合每个GraphFormer最后输出的GNN的输出
        self.aggregate_LastGNN = nn.Linear(self.block_num * self.mem_dim, self.mem_dim)
        # 聚合每个GraphFormer中Transformer和最后的GNN的输出
        self.aggregate_GLF = nn.Linear(self.block_num * 2 * self.mem_dim, self.mem_dim)
        # 聚合每个GraphFormer中Transformer和所有GNN的输出
        self.aggregate_GAF = nn.Linear(self.block_num * (1 + self.num_layers) * self.mem_dim, self.mem_dim)

    def forward(self, adj, inputs):
        words, masks, pos, deprel, head, first_pos, second_pos = inputs  # unpack
        src_mask = (words != constant.PAD_ID).unsqueeze(-2)

        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs) # (bs,seq_len, in_dim)
        outputs = self.input_W_R(embs)  # in_dim -> mem_dim
        # ************************** 以上outputs部分是原始输入，下面开始执行GraphFormer的核心模块 **************************

        LastGNN_layers_list = []  # 用于存储每个graphformer block最后一层的输出
        GLF_layers_list = [] # 聚合每个GraphFormer中Transformer和最后的GNN的输出
        GAF_layers_list = [] # 聚合每个GraphFormer中Transformer和所有GNN的输出
        # print(f"---------------------------------------------------------{self.block_num}")
        # -------------------------- 该部分内容为每个GraphFormer block中，transformer放在gcn的前面 ----------------------
        # 此处开始添加多层transformer和GCN构建的graphformer
        if self.module_sequence_selector == 0:
            # print("Transformer placed before GCN")
            for gf_i in range(self.block_num):
                gf_layer = self.graphformer_layers[gf_i] # 获取一个graphformer层
                net_layers = len(gf_layer)
                assert net_layers == self.num_layers * 2
                # transformer, (bs,seq_len, mem_dim) -> (bs,seq_len, mem_dim)，words只用来进行mask的获取，不代表rep的实际使用
                outputs, _ = gf_layer[0](outputs, words)
                GAF_layers_list.append(outputs)
                GLF_layers_list.append(outputs)

                # 下面执行GNN部分。对于1执行初始层；接下来，每两层开始执行
                temp_last_gnn_outputs = None
                for gnn_i in range(1, net_layers, 2):
                    if gnn_i == 1:
                        outputs = gf_layer[gnn_i](adj, outputs)  # mem_dim, GraphConvLayer
                        GAF_layers_list.append(outputs)
                    else: # 3,5,7,...
                        attn_tensor = gf_layer[gnn_i-1](outputs, outputs, src_mask) # MultiHeadAttention
                        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                        outputs = gf_layer[gnn_i](attn_adj_list, outputs)  # mem_dim
                        GAF_layers_list.append(outputs)
                    temp_last_gnn_outputs = outputs
                GLF_layers_list.append(temp_last_gnn_outputs)
                LastGNN_layers_list.append(temp_last_gnn_outputs)
        # ----------------------------- 该部分内容为每个GraphFormer block中，transformer放在gcn的前面 -------------------

        # ----------------------------- 该部分内容为每个GraphFormer block中，transformer放在gcn的后面 -------------------

        # 此处开始添加多层transformer和GCN构建的graphformer
        else:
            # print("Transformer placed after GCN")
            for gf_i in range(self.block_num):
                gf_layer = self.graphformer_layers[gf_i] # 获取一个graphformer层
                net_layers = len(gf_layer)
                assert net_layers == self.num_layers * 2

                # 下面执行GNN部分。对于0执行初始层；接下来，每两层开始执行
                temp_last_gnn_outputs = None
                for gnn_i in range(0, net_layers-1, 2):
                    if gnn_i == 0:
                        outputs = gf_layer[gnn_i](adj, outputs)  # mem_dim, GraphConvLayer
                        GAF_layers_list.append(outputs)
                    else: # 3,5,7,...
                        attn_tensor = gf_layer[gnn_i-1](outputs, outputs, src_mask) # MultiHeadAttention
                        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                        outputs = gf_layer[gnn_i](attn_adj_list, outputs)  # mem_dim
                        GAF_layers_list.append(outputs)
                    temp_last_gnn_outputs = outputs
                GLF_layers_list.append(temp_last_gnn_outputs)
                LastGNN_layers_list.append(temp_last_gnn_outputs)

            # transformer, (bs,seq_len, mem_dim) -> (bs,seq_len, mem_dim)，words只用来进行mask的获取，不代表rep的实际使用
            outputs, _ = gf_layer[net_layers-1](outputs, words)
            GAF_layers_list.append(outputs)
            GLF_layers_list.append(outputs)


        # ----------------------------- 该部分内容为每个GraphFormer block中，transformer放在gcn的后面 -------------------

        # 在下面三者之中选择一个
        if self.aggregator == 0:
            # print("aggregator == 0")
            aggregate_LastGNN_outputs = torch.cat(LastGNN_layers_list, dim=2)
            dcgcn_output = self.aggregate_LastGNN(aggregate_LastGNN_outputs)
        elif self.aggregator == 1:
            # print("aggregator == 1")
            aggregate_GLF_outputs = torch.cat(GLF_layers_list, dim=2)
            dcgcn_output = self.aggregate_GLF(aggregate_GLF_outputs)
        else:
            # print("aggregator == 2")
            aggregate_GAF_outputs = torch.cat(GAF_layers_list, dim=2)
            dcgcn_output = self.aggregate_GAF(aggregate_GAF_outputs)

        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        return dcgcn_output, mask


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn
