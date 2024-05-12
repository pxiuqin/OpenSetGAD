import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl

torch.set_default_dtype(torch.float32)

class GateLayer(nn.Module):
    def __init__(self, emb_size, emb_dropout=0):
        super(GateLayer, self).__init__()
        self.gating_weight = nn.Parameter(torch.empty(emb_size, emb_size))
        self.gating_bias = nn.Parameter(torch.empty(1, emb_size))

        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        return self.emb_dropout(torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias)))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False, phase='pretrain'):
        super(GATLayer, self).__init__()

        # 更加phase来判断是否为finetune
        # if phase == 'finetune':
        #     self.emb_gate = GateLayer(in_dim)
        # else:
        #     self.emb_gate = None

        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    # def message_func(self, edges):
    #     # message UDF for equation (3) & (4)
    #     return {'z': edges.src['z'], 'e': edges.data['e']}

    # def reduce_func(self, nodes):
    #     # reduce UDF for equation (3) & (4)
    #     # equation (3)
    #     alpha = F.softmax(nodes.mailbox['e'], dim=1)
    #     # equation (4)
    #     h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    #     return {'h': h}
    
    # def edge_attention(self, edges):
    #     # edge UDF for equation (2)
    #     z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    #     a = self.attn_fc(z2)
    #     # 获取边的时间权重
    #     time_weight = edges.data['time_weight']
    #     # 将时间权重与注意力分数相乘
    #     a = a * time_weight.unsqueeze(-1)
    #     return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        # 传递时间权重
        return {'z': edges.src['z'], 'e': edges.data['e'], 'time_weight': edges.data['time_weight'].unsqueeze(1)}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 获取时间权重
        time_weight = F.softmax(nodes.mailbox['time_weight'], dim=1)
        # equation (4)
        # 使用时间权重加权邻居节点的特征
        # h = torch.sum((alpha/2 + time_weight/2)* nodes.mailbox['z'], dim=1)
        h = torch.sum((alpha*0.4 + time_weight*0.6)* nodes.mailbox['z'], dim=1)
        return {'h': h}
    
    def forward(self, blocks, layer_id):
        h = blocks[layer_id].srcdata['features']
        # h = self.emb_gate(h) if self.emb_gate else h
        z = self.fc(h)
        blocks[layer_id].srcdata['z'] = z
        z_dst = z[:blocks[layer_id].number_of_dst_nodes()]
        blocks[layer_id].dstdata['z'] = z_dst
        blocks[layer_id].apply_edges(self.edge_attention)
        # equation (3) & (4)
        blocks[layer_id].update_all(  # block_id – The block to run the computation.
                         self.message_func,  # Message function on the edges.
                         self.reduce_func)  # Reduce function on the node.

        # nf.layers[layer_id].data.pop('z')
        # nf.layers[layer_id + 1].data.pop('z')

        if self.use_residual:
            return z_dst + blocks[layer_id].dstdata['h']  # residual connection
        return blocks[layer_id].dstdata['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False, phase='pretrain'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual, phase))
        self.merge = merge

    def forward(self, blocks, layer_id):
        head_outs = [attn_head(blocks, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False, phase='pretrain'):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual, phase)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual, phase)  # 一个Head做输出
    
    def forward(self, blocks):
        h = self.layer1(blocks, 0)
        h = F.elu(h)
        # print(h.shape)
        blocks[1].srcdata['features'] = h  # 把第0层的输出来更新第1层的src features
        h = self.layer2(blocks, 1)
        h = F.normalize(h, p=2, dim=1)

        return h  # 这里输出第1层的 features  维度为 out_dim 默认64
    
class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        # uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn
    
class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GIN, self).__init__()

        self.act=torch.nn.ReLU()
        self.g_net, self.bns, g_dim = self.create_net(
            name="graph", input_dim=in_dim,hidden_dim=hidden_dim)
        self.num_layers_num=3
        self.dropout=torch.nn.Dropout(p=0.5)


    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        num_rels = kw.get("num_rels", 1)
        num_bases = kw.get("num_bases", 8)
        regularizer = kw.get("regularizer", "basis")
        dropout = kw.get("dropout", 0.5)


        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), self.act, torch.nn.Linear(hidden_dim, hidden_dim))
            else:
                nn = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), self.act, torch.nn.Linear(hidden_dim, hidden_dim))
            conv = dgl.nn.pytorch.conv.GINConv(apply_func=nn,aggregator_type='sum')
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns, hidden_dim


    #def forward(self, pattern, pattern_len, graph, graph_len):
    def forward(self, graph, graph_len,graphtask=False):
        graph_output = graph.ndata["feature"]
        xs = []
        for i in range(self.num_layers_num):
            graph_output = F.relu(self.convs[i](graph,graph_output))
            graph_output = self.bns[i](graph_output)
            graph_output = self.dropout(graph_output)
            xs.append(graph_output)
        xpool= []
        for x in xs:
            if graphtask:
                graph_embedding = self.split_and_batchify_graph_feats(x, graph_len)[0]
            else:
                graph_embedding=x
            graph_embedding = torch.sum(graph_embedding, dim=1)
            xpool.append(graph_embedding)
        x = torch.cat(xpool, -1)
        #x is graph level embedding; xs is node level embedding
        return x,torch.cat(xs, -1)
    
    def split_and_batchify_graph_feats(self, batched_graph_feats, graph_sizes):
        bsz = graph_sizes.size(0)
        dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

        min_size, max_size = graph_sizes.min(), graph_sizes.max()
        mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

        if min_size == max_size:
            return batched_graph_feats.view(bsz, max_size, -1), mask
        else:
            graph_sizes_list = graph_sizes.view(-1).tolist()
            unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
            for i, l in enumerate(graph_sizes_list):
                if l == max_size:
                    continue
                elif l > max_size:
                    unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
                else:
                    mask[i, l:].fill_(0)
                    zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
                    unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
            return torch.stack(unbatched_graph_feats, dim=0), mask

