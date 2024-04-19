import gc
import math
import time
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# 用于GraphPrompt框架用
def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
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
    
#use prompt to finish step 1
class graph_prompt_layer_mean(nn.Module):
    def __init__(self):
        super(graph_prompt_layer_mean, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(2, 2))
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result

class node_prompt_layer_linear_mean(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(node_prompt_layer_linear_mean, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)

    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        return graph_embedding

class node_prompt_layer_linear_sum(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(node_prompt_layer_linear_sum, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)

    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        # print(graph_embedding)
        return graph_embedding



#sum result is same as mean result
class node_prompt_layer_sum(nn.Module):
    def __init__(self):
        super(node_prompt_layer_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(2, 2))
    def forward(self, graph_embedding, graph_len):
        return graph_embedding



class graph_prompt_layer_weighted(nn.Module):
    def __init__(self,max_n_num):
        super(graph_prompt_layer_weighted, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight[0][0:graph_embedding.size(1)]
        temp1 = torch.ones(graph_embedding.size(0), graph_embedding.size(2), graph_embedding.size(1)).to(graph_embedding.device)
        temp1 = weight * temp1
        temp1 = temp1.permute(0, 2, 1)
        graph_embedding=graph_embedding*temp1
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class node_prompt_layer_feature_weighted_mean(nn.Module):
    def __init__(self,input_dim):
        super(node_prompt_layer_feature_weighted_mean, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=graph_embedding*self.weight
        return graph_embedding

class node_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,input_dim):
        super(node_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        print('reset--------------------')
        torch.nn.init.xavier_uniform_(self.weight)
        # print(self.weight)
    def forward(self, graph_embedding, graph_len):
        # print(self.weight)
        graph_embedding=graph_embedding*self.weight
        print(graph_embedding)
        return graph_embedding

class graph_prompt_layer_weighted_matrix(nn.Module):
    def __init__(self,max_n_num,input_dim):
        super(graph_prompt_layer_weighted_matrix, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(input_dim,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight.permute(1, 0)[0:graph_embedding.size(1)]
        weight = weight.expand(graph_embedding.size(0), weight.size(0), weight.size(1))
        graph_embedding = graph_embedding * weight
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class graph_prompt_layer_weighted_linear(nn.Module):
    def __init__(self,max_n_num,input_dim,output_dim):
        super(graph_prompt_layer_weighted_linear, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,max_n_num))
        self.linear=nn.Linear(input_dim,output_dim)
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight[0][0:graph_embedding.size(1)]
        temp1 = torch.ones(graph_embedding.size(0), graph_embedding.size(2), graph_embedding.size(1)).to(graph_embedding.device)
        temp1 = weight * temp1
        temp1 = temp1.permute(0, 2, 1)
        graph_embedding=graph_embedding*temp1
        graph_prompt_result = graph_embedding.mean(dim=1)
        return graph_prompt_result

class graph_prompt_layer_weighted_matrix_linear(nn.Module):
    def __init__(self,max_n_num,input_dim,output_dim):
        super(graph_prompt_layer_weighted_matrix_linear, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(output_dim,max_n_num))
        self.linear=nn.Linear(input_dim,output_dim)
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight.permute(1, 0)[0:graph_embedding.size(1)]
        weight = weight.expand(graph_embedding.size(0), weight.size(0), weight.size(1))
        graph_embedding = graph_embedding * weight
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result

def bp_compute_abmae(predict, count):
    error = torch.absolute(predict-count)/(count+1)
    return error.mean()

def distance2center(input,center):
    n = input.size(0)
    m = input.size(1)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    #print('input power: ',input_power)
    center_power = torch.sum(center * center, dim=1).expand(n, k)
    #print('center power: ',center_power)
    temp1=input_power+center_power
    temp2=2*torch.mm(input,center.transpose(0,1))
    #print('input power+ center power: ',temp1)
    #print('2*mm(input,center): ',temp2)
    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance

# THIS WILL NOT RETURN NEGATIVE VALUE
def distance2center2(input,center):
    n = input.size(0)
    m = input.size(1)
    k = center.size(0)
    input_expand = input.reshape(n, 1, m).expand(n, k, m)
    center_expand = center.expand(n, k, m)
    temp = input_expand - center_expand
    temp = temp * temp
    distance = torch.sum(temp, dim=2)
    return distance


def center_embedding(input,node_labels,label_num=0,debug=False):
    node_labels = node_labels.unsqueeze(1)
    device=input.device
    label_num = torch.max(node_labels) + 1
    mean = torch.ones(node_labels.size(0), node_labels.size(1)).to(device)
    index = node_labels  # torch.tensor(node_labels, dtype=int).to(device)
    # mean = torch.ones(label_num, 1).to(device)
    # index = torch.tensor(node_labels,dtype=int).to(device)
    # index = torch.unsqueeze(index, dim=0)

    if debug:
        print(node_labels)

    _mean = torch.zeros(label_num, 1, device=device).scatter_add_(dim=0, index=index, src=mean)
    preventnan = torch.ones(_mean.size(), device=device)*0.0000001
    _mean = _mean+preventnan
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    c = c / _mean
    return c

def anneal_fn(fn, t, T, lambda0=0.0, lambda1=1.0):
    if not fn or fn == "none":
        return lambda1
    elif fn == "logistic":
        K = 8 / T
        return float(lambda0 + (lambda1-lambda0)/(1+np.exp(-K*(t-T/2))))
    elif fn == "linear":
        return float(lambda0 + (lambda1-lambda0) * t/T)
    elif fn == "cosine":
        return float(lambda0 + (lambda1-lambda0) * (1 - math.cos(math.pi * t/T))/2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return anneal_fn(fn.split("_", 1)[1], t-R*T, R*T, lambda1, lambda0)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return lambda1
    else:
        raise NotImplementedError
    
def correctness_GPU(pre, counts):
    temp=pre-counts
    #print(temp.size())
    nonzero_num=torch.count_nonzero(temp)
    return (len(temp)-nonzero_num)/len(temp)

def correctness(pred,counts):
    return accuracy_score(counts,pred)

'''def f1score(pred,counts):
    return f1score(counts,pred)'''

def microf1(pred,counts):
    #pre=precision_score(counts,pred,average='micro')
    #re=recall_score(counts,pred,average='micro')
    #return 2*pre*re/(pre+re)
    return f1_score(counts,pred,average='micro')

def macrof1(pred,counts):
    #pre=precision_score(counts,pred,average='micro')
    #re=recall_score(counts,pred,average='micro')
    #return 2*pre*re/(pre+re)
    return f1_score(counts,pred,average='macro')

def weightf1(pred,counts):
    return f1_score(counts,pred,average='weighted')

def calc_loss(embedding, device, config, node_label):
    if config.reg_loss == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config.reg_loss == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config.reg_loss == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config.reg_loss == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config.reg_loss == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.leaky_relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred), target)
    else:
        raise NotImplementedError

    if config.bp_loss == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config.bp_loss == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config.bp_loss == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config.bp_loss == "NLL":
        bp_crit = lambda pred, target, neg_slp: F.nll_loss(pred, target)
    elif config.bp_loss=="CROSS":
        bp_crit = lambda pred, target, neg_slp: F.cross_entropy(pred, target)
    elif config.bp_loss == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    node_label -= 1  # 转换成那种index的形式
    c_embedding=center_embedding(embedding,node_label)
    distance=distance2center2(embedding,c_embedding)
    #print(distance)

    distance = 1/F.normalize(distance, dim=1)

    #distance=distance2center2(embedding,c_embedding)

    #print('distance: ',distance )
    pred = F.log_softmax(distance, dim=1)
        
    # 选择正确类别的log softmax值
    correct_log_softmax = pred.gather(1, node_label.unsqueeze(1)).squeeze(1)

    _pred = torch.argmax(pred, dim=1, keepdim=True).squeeze()
    accuracy = correctness_GPU(_pred, node_label)

    #reg_loss = reg_crit(pred, graph_label_onehot)
    #对NLL LOSS用这个公式，否则用上面的
    # reg_loss = reg_crit(_pred, node_label.to(device))
    # reg_loss.requires_grad_(True)

    # if isinstance(config.bp_loss_slp, (int, float)):
    #     neg_slp = float(config.bp_loss_slp)
    # else:
    #     bp_loss_slp, l0, l1 = config.bp_loss_slp.rsplit("$", 3)
    #     neg_slp = anneal_fn(bp_loss_slp, t=1, T=10, lambda0=float(l0),
    #                         lambda1=float(l1))

    # bp_loss = bp_crit(correct_log_softmax.float(), node_label.to(device), neg_slp)
    
    # 计算损失
    bp_loss = -correct_log_softmax.mean()

    return bp_loss, accuracy

def prompt_loss(node_features, node_label, class_feature=None, temperature=1.0):
    """
    计算prompt损失函数。
    
    :param node_features: 节点特征表示的张量，形状为 (batch_size, num_features)
    :param class_feature: 类别原型子图表示的张量，形状为 (num_classes, num_features)
    :param node_label: 实例的真实类别标签的张量，形状为 (batch_size,)
    :param temperature: 温度参数，用于控制softmax的平滑程度
    :return: 计算得到的prompt损失值
    """    

    # node_label -= 1  # 转换成那种index的形式
    if class_feature is not None:
        c_embedding = class_feature
    else:
        c_embedding = center_embedding(node_features, node_label)
    distance = distance2center2(node_features, c_embedding)

    distance = 1/F.normalize(distance, dim=1)
    # distance /= temperature   # 应用温度参数  

    pred = F.log_softmax(distance, dim=1)
    _pred = torch.argmax(pred, dim=1, keepdim=True).squeeze()
    
    # 计算log softmax
    log_softmax = F.log_softmax(pred, dim=1)
    
    # 选择正确类别的log softmax值
    correct_log_softmax = log_softmax.gather(1, node_label.unsqueeze(1)).squeeze(1)
    
    # 计算损失
    loss = -correct_log_softmax.mean()
    
    return loss, _pred


class label2onehot(torch.nn.Module):
    def __init__(self, labelnum,device):
        super(label2onehot, self).__init__()
        self.labelnum=labelnum
        self.device=device
    def forward(self,input):
        labelnum = torch.tensor(self.labelnum).to(self.device)
        index=torch.tensor(input,dtype=int).to(self.device)
        output = torch.zeros(input.size(0), labelnum).to(self.device)
        src = torch.ones(input.size(0), labelnum).to(self.device)
        output = torch.scatter_add(output, dim=1, index=index, src=src)
        return output

def evaluate(model, data_type, device, config, epoch, c_embedding, label_num, pretrain_embedding, node_label, debug=False,logger=None, writer=None):
    epoch_step = len(data_loader)
    total_reg_loss = 0
    total_step = config["epochs"] * epoch_step
    total_bp_loss = 0
    batchcnt=0
    total_acc=0
    total_macrof=0
    total_weighted=0
    total_cnt = 1e-6

    evaluate_results = {"data": {"id": list(), "counts": list(), "pred": list()},
                        "error": {"mae": INF, "mse": INF},
                        "time": {"avg": list(), "total": 0.0}}

    if config.reg_loss == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target, reduce="none")
    elif config.reg_loss == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target, reduce="none")
    elif config.reg_loss == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target, reduce="none")
    elif config.reg_loss == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config.reg_loss == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred), target,
                                                                                                 reduce="none")
    else:
        raise NotImplementedError

    if config.bp_loss == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config.bp_loss == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config.bp_loss == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config.bp_loss == "NLL":
        bp_crit = lambda pred, target,neg_slp: F.nll_loss(pred, target)
    elif config.bp_loss == "CROSS":
        bp_crit = lambda pred, target, neg_slp: F.cross_entropy(pred, target)
    elif config.bp_loss == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.eval()
    l2onehot=label2onehot(train_config["graph_label_num"],device)
    label_num=torch.tensor(label_num).to(device)
    total_time = 0
    batchcnt+=1

    s = time.time()
    # if debug:
    #     print("####################")
    #     print("pretrain embedding:",embedding)
    embedding = model(pretrain_embedding, 0)*train_config["scalar"]
    node_label=node_label
    c_embedding = center_embedding(embedding, node_label, label_num,debug)

    distance=distance2center(embedding,c_embedding)
    distance=-1*F.normalize(distance,dim=1)

    pred=F.log_softmax(distance,dim=1)
    # if debug:
    #     print("pred:",pred)

    reg_loss = reg_crit(pred, node_label.squeeze().type(torch.LongTensor).to(device))

    if isinstance(config["bp_loss_slp"], (int, float)):
        neg_slp = float(config["bp_loss_slp"])
    else:
        bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
        # neg_slp = anneal_fn(bp_loss_slp, 0 + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
        #                     lambda1=float(l1))
        neg_slp=0.2
    bp_loss = bp_crit(pred, node_label.squeeze().type(torch.LongTensor).to(device), neg_slp)

    #graph_label_onehot=l2onehot(graph_label)
    _pred = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(_pred, node_label)
    eval_pred=_pred.cpu().numpy()
    eval_graph_label=node_label.cpu().numpy()
    acc=correctness(eval_pred,eval_graph_label)
    macrof=macrof1(eval_pred,eval_graph_label)
    weightf=weightf1(eval_pred,eval_graph_label)
    total_acc+=acc
    total_macrof+=macrof
    total_weighted+=weightf


    # float
    reg_loss_item = reg_loss.item()
    bp_loss_item = bp_loss.item()
    if writer:
        writer.add_scalar("%s/REG-%s" % (data_type, config.reg_loss), reg_loss_item,
                          epoch * epoch_step + 0)
        writer.add_scalar("%s/BP-%s" % (data_type, config.bp_loss), bp_loss_item,
                          epoch * epoch_step + 0)

    if logger and 0 == epoch_step - 1:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\taccuracy: {:0>1.3f}".format(
                epoch, config["epochs"], data_type, 0, epoch_step,
                reg_loss_item, bp_loss_item,accuracy))
    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config.reg_loss), reg_loss.item(), epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config.bp_loss), bp_loss.item(), epoch)
    if logger:
        logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\tacc:{:0>1.3f}".format(
            epoch, config["epochs"], data_type, reg_loss_item, bp_loss_item,acc))


    gc.collect()
    #return mean_reg_loss, mean_bp_loss, evaluate_results, total_time,mean_acc.cpu(),macrof,weightedf,c_embedding
    return mean_reg_loss, mean_bp_loss, evaluate_results, total_time,acc,macrof,weightf,c_embedding
