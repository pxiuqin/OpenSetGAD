# from domain_aware_prompt import DAPromptHead
import joblib
from graph_pro_prompt import GraphPro
from graph_prompt import center_embedding, distance2center2, node_prompt_layer_feature_weighted_mean, node_prompt_layer_feature_weighted_sum, node_prompt_layer_linear_mean, node_prompt_layer_linear_sum, node_prompt_layer_sum, calc_loss, prompt_loss, prompt_loss_with_center
from load_data import getdata
from model_dynamic import GAT
# from model import GAT
import torch.optim as optim
import time
import numpy as np
import torch
import os
import dgl
from sklearn import metrics
from sklearn.cluster import KMeans
from utils import pairwise_sample
from utils import edl_digamma_loss, relu_evidence, edl_mse_loss, edl_log_loss
import torch.nn.functional as F
from load_data import SocialDataset
from gpf_prompt import CenterEmbedding, SimplePrompt, GPFplusAtt, embeddings

INF = float("inf")

MAX_SCORE = {'1':0.43,'2':0.81,'3':0.78,'4':0.71,'5':0.75,'6':0.83,'7':0.57,'8':0.80,'9':0.77,'10':0.82,'11':0.75,'12':0.70,'13':0.68,'14':0.69,'15':0.59,'16':0.79,'17':0.71,'18':0.7,'19':0.73,'20':0.73,'21':0.61}
# MAX_SCORE = {'1':0.57,'2':0.58,'3':0.57,'4':0.58,'5':0.61,'6':0.60,'7':0.64,'8':0.58,'9':0.52,'10':0.60,'11':0.60,'12':0.61,'13':0.60,'14':0.68,'15':0.63,'16':0.51}

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def process_tensor(tensor, labels):
    result = torch.ones_like(tensor)
    for i, value in enumerate(tensor):
        if value.item() in labels:
            result[i] = 0

    # 统计值为 1 的个数
    count_ones = torch.sum(result == 1)

    # 统计值为 0 的个数
    count_zeros = torch.sum(result == 0)

    print(f"The number of 1's: {count_ones}")
    print(f"The number of 0's: {count_zeros}")

    return result

def run_kmeans(extract_features, extract_labels, indices, args,isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)   # 因为是互信息所以label标识不必一致
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:",nmi,'ami:',ami,'ari:',ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics =='ari':
        print('use ari')
        value = ari
    if args.metrics=='ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)

def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, args, is_validation=True):
    epoch +=1   # 让其从1开始
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, args)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode +' '
    message += args.metrics +': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value= run_kmeans(extract_features, extract_labels, indices, args,
                                              save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' value: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI))
    print(message)

    all_value_save_path = "/".join(save_path.split('/')[0:-1])
    print(all_value_save_path)

    with open(all_value_save_path + '/evaluate.txt', 'a') as f:
        f.write("block "+ save_path.split('/')[-1])
        f.write(message)
        f.write('\n')
        f.write("NMI "+str(NMI)+" AMI "+str(AMI) + ' ARI '+str(ARI) + '\n')

    # 如果在测试模型，把模型的结果保存为csv数据
    block_path = save_path + '/evaluate.csv'
    local_value = f"\n{epoch},{NMI},{AMI},{ARI}"
    if os.path.exists(block_path):
        with open(block_path, 'a') as f:
            f.write(local_value)
    else:
        with open(block_path, 'w') as f:
            f.write(f"Epoch,NMI,AMI,ARI")
            f.write(local_value)

    all_path = all_value_save_path + '/evaluate.csv'
    global_value = f"\n{save_path.split('/')[-1]},{epoch},{NMI},{AMI},{ARI}"
    if os.path.exists(all_path):
        with open(all_path, 'a') as f:
            f.write(global_value)
    else:
        with open(all_path, 'w') as f:
            f.write(f"Block,Epoch,NMI,AMI,ARI")
            f.write(global_value)

    return value

def extract_embeddings(g, model, num_all_samples, args, prompt = None, prompt_model = None):
    with torch.no_grad():
        model.eval()
        if prompt:
            prompt_model.eval()

        indices = torch.LongTensor(np.arange(0,num_all_samples,1))
        if args.use_cuda:
            indices = indices.cuda()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, graph_sampler=sampler,
            batch_size=num_all_samples,
            indices = indices,
            shuffle=False,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            extract_labels = blocks[-1].dstdata['labels']
            if prompt == 'gpf':
                # blocks[0].srcdata['features'] = prompt_model.add(blocks[0].srcdata['features'])
                blocks = embeddings(prompt_model, blocks)
                extract_features = model(blocks)
            elif prompt == 'graph_prompt':
                extract_features = model(blocks)
                extract_features = prompt_model(extract_features, 0)
            elif prompt == 'graphpro':
                blocks[0].srcdata['features'] = prompt_model(blocks[0].srcdata['features'])
                extract_features = model(blocks)
            elif prompt == 'prompt':
                blocks[0].srcdata['features'] = prompt_model.add(blocks[0].srcdata['features'])
                extract_features = model(blocks)
                # extract_features = prompt_model.add(extract_features)
            else:
                extract_features = model(blocks)

        assert batch_id == 0
        if torch.any(torch.isnan(extract_features)):
            print('-------------------------------')

        extract_features = extract_features.data.cpu().numpy()
        extract_labels = extract_labels.data.cpu().numpy()

    return (extract_features, extract_labels)

def initial_train(i, args, data_split, metrics,embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices, adjacency_matrix = getdata(
        embedding_save_path, data_split, i, args)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    if args.use_cuda:
        model.cuda()

    # print(model.parameters())
    # Optimizer
    optimizer = optim.Adam(params=model.parameters(),lr=args.lr, weight_decay=1e-4)

    # Start training
    message = "\n------------ Start initial training ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_value = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_value = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        label_center = {}
        for l in set(extract_labels):
            l_indices = np.where(extract_labels==l)[0]
            l_feas = extract_features[l_indices]
            l_cen = np.mean(l_feas,0)
            label_center[l] = l_cen


        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)  # 两层图结构
        dataloader = dgl.dataloading.DataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )


        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']  # 最后一个块输出的标签

            start_batch = time.time()
            model.train()
            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).


            loss_outputs = loss_fn(pred, batch_labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            dis = torch.empty([0, 1]).cuda() if args.use_cuda else torch.empty([0, 1])
            for l in set(batch_labels.cpu().data.numpy()):
                label_indices = torch.where(batch_labels==l)
                l_center = torch.FloatTensor(label_center[l]).cuda() if args.use_cuda else torch.FloatTensor(label_center[l])
                dis_l = (pred[label_indices] - l_center).pow(2).sum(1).unsqueeze(-1)
                dis = torch.cat([dis,dis_l],0)

            # 这里注意，如果启用了pairwise lose，就注释掉上面的loss定义
            if args.add_pair:
                pairs, pair_labels, pair_matrix = pairwise_sample(pred, batch_labels)
                if args.use_cuda:
                    pairs = pairs.cuda()
                    pair_matrix = pair_matrix.cuda()
                    # pair_labels = pair_labels.unsqueeze(-1).cuda()

                pos_indices = torch.where(pair_labels > 0)
                neg_indices = torch.where(pair_labels == 0)   # 不配对为负样本

                # 确定范围从0到neg_indices的个数，然后让数量扩展到5倍的正样本数
                neg_ind = torch.randint(0, neg_indices[0].shape[0], [5*pos_indices[0].shape[0]]).cuda() if args.use_cuda else torch.randint(0, neg_indices[0].shape[0], [5*pos_indices[0].shape[0]])
                neg_dis = (pred[pairs[neg_indices[0][neg_ind], 0]] - pred[pairs[neg_indices[0][neg_ind], 1]]).pow(2).sum(1).unsqueeze(-1)
                pos_dis = (pred[pairs[pos_indices[0], 0]] - pred[pairs[pos_indices[0], 1]]).pow(2).sum(1).unsqueeze(-1)
                pos_dis = torch.cat([pos_dis]*5,0)  # 这里为什么要复制5份，因为要和上面的5倍匹配
                pairs_indices = torch.where(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)>0)  # 必须严格大于0
                loss = torch.mean(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)[pairs_indices[0]])   # 对距离求平均

                label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
                pred = F.normalize(pred, 2, 1)
                pair_out = torch.mm(pred,pred.t())  # H*H^t  结果为d*d的矩阵

                # 是否开启正交
                if args.add_ort:
                    pair_loss = (pair_matrix - pair_out).pow(2).mean()
                    print("pair loss:",loss,"pair orthogonal loss:  ",100*pair_loss)
                    loss += 100 * pair_loss
       
            losses.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
        np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)

        # 这里的值是NMI或者AMI
        validation_value = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)
        all_vali_value.append(validation_value)

        # Early stop
        if validation_value > best_vali_value:
            best_vali_value = validation_value
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)

        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    np.save(save_path_i + '/all_vali_value.npy', np.asarray(all_vali_value))
    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')
    # Load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")


    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    label_center = {}

    # 变量不同label，并计算该label对应的平均特征表示
    for l in set(extract_labels):
        l_indices = np.where(extract_labels == l)[0]
        l_feas = extract_features[l_indices]
        l_cen = np.mean(l_feas, 0)
        label_center[l] = l_cen
    joblib.dump(label_center,save_path_i + '/models/label_center.dump')
    
    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    torch.save(label_center_emb,save_path_i + '/models/center.pth')

    if args.add_pair:
        return model, label_center_emb
    else:
        return model

def initial_train_with_prompt_loss(i, args, data_split, metrics, embedding_save_path, class_emb, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices, adjacency_matrix = getdata(
        embedding_save_path, data_split, i, args)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    if args.use_cuda:
        model.cuda()

    # print(model.parameters())
    # Optimizer
    optimizer = optim.Adam(params=model.parameters(),lr=args.lr, weight_decay=1e-4)

    # Start training
    message = "\n------------ Start initial training ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_value = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_value = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)  # 两层图结构
        dataloader = dgl.dataloading.DataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )


        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']  # 最后一个块输出的标签

            start_batch = time.time()
            model.train()
            
            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
            loss_outputs,_pred = prompt_loss(pred, batch_labels, adjacency_matrix)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(_pred, batch_labels, loss_outputs)

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
        np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)

        # 这里的值是NMI或者AMI
        validation_value = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)
        all_vali_value.append(validation_value)

        # Early stop
        if validation_value > best_vali_value:
            best_vali_value = validation_value
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)

        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    np.save(save_path_i + '/all_vali_value.npy', np.asarray(all_vali_value))
    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')
    # Load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")


    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    label_center = {}

    # 变量不同label，并计算该label对应的平均特征表示
    for l in set(extract_labels):
        l_indices = np.where(extract_labels == l)[0]
        l_feas = extract_features[l_indices]
        l_cen = np.mean(l_feas, 0)
        label_center[l] = l_cen
    joblib.dump(label_center,save_path_i + '/models/label_center.dump')

    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    torch.save(label_center_emb,save_path_i + '/models/center.pth')

    if args.add_pair:
        return model, label_center_emb
    else:
        return model

def initial_train_with_prompt_loss_new(i, args, data_split, metrics, embedding_save_path, class_emb, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices, adjacency_matrix = getdata(
        embedding_save_path, data_split, i, args)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    if args.use_cuda:
        model.cuda()

    prompt = CenterEmbedding(args.out_dim, len(set(labels.tolist())))

    model_param_group = []
    model_param_group.append({"params": model.parameters()})
    model_param_group.append({"params": prompt.parameters()})

    # print(model.parameters())
    # Optimizer
    optimizer = optim.Adam(params=model_param_group,lr=args.lr, weight_decay=1e-4)

    # Start training
    message = "\n------------ Start initial training ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_value = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_value = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)  # 两层图结构
        dataloader = dgl.dataloading.DataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )


        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']  # 最后一个块输出的标签

            start_batch = time.time()
            model.train()
            prompt.train()
            
            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
            c_embedding_prompt = prompt(pred, batch_labels)

            loss,_pred = prompt_loss_with_center(pred, batch_labels, c_embedding_prompt, adjacency_matrix)
            losses.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(_pred, batch_labels, loss)

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        # np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
        # np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)

        # 这里的值是NMI或者AMI
        validation_value = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)
        all_vali_value.append(validation_value)

        # Early stop
        if validation_value > best_vali_value:
            best_vali_value = validation_value
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)

        # else:
        #     wait += 1
        # if wait == args.patience:
        #     print('Saved all_mins_spent')
        #     print('Early stopping at epoch ', str(epoch))
        #     print('Best model was at epoch ', str(best_epoch))
        #     break
        # end one epoch

    # Save all validation nmi
    # np.save(save_path_i + '/all_vali_value.npy', np.asarray(all_vali_value))
    # Save time spent on epochs
    # np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    # print('Saved mins_train_epochs.')
    # Save time spent on batches
    # np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    # print('Saved seconds_train_batches.')
    # Load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")


    label_center = {}
    with torch.no_grad():
        prompt.eval()
        c_embedding_prompt = prompt(torch.FloatTensor(extract_features), torch.LongTensor(extract_labels))

    # 变量不同label，并计算该label对应的平均特征表示
    for l in set(extract_labels):
        # 采用求平均的方法
        # l_indices = np.where(extract_labels == l)[0]
        # l_feas = extract_features[l_indices]
        # l_cen = np.mean(l_feas, 0)
        # label_center[l] = l_cen

        # 采用CenterEmbedding的方式
        label_center[l] = c_embedding_prompt[l]

    joblib.dump(label_center,save_path_i + '/models/label_center.dump')

    # label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    # torch.save(label_center_emb,save_path_i + '/models/center.pth')

    return model


def continue_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center_emb, args):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    # 这里是个假命题，我感觉，可能是i等于0时，执行这里
    if i%1!=0:
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, 0, num_isolated_nodes,
                              save_path_i, args, True)
        return model


    else:
        # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                              save_path_i, args, True)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Start fine tuning
        message = "\n------------ Start fine tuning ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

        # record the time spent in seconds on each batch of all training/maintaining epochs
        seconds_train_batches = []
        # record the time spent in mins on each epoch
        mins_train_epochs = []
        for epoch in range(args.finetune_epochs):
            start_epoch = time.time()
            losses = []
            total_loss = 0
            for metric in metrics:
                metric.reset()

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.DataLoader(
                g, test_indices, sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                )

            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
                blocks = [b.to(device) for b in blocks]
                batch_labels = blocks[-1].dstdata['labels']

                start_batch = time.time()
                model.train()
                label_center_emb.to(device)

                # forward
                pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
                pred = F.normalize(pred, 2, 1)
                rela_center_vec = torch.mm(pred,label_center_emb.t())
                rela_center_vec = F.normalize(rela_center_vec,2,1)
                entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
                entropy = torch.sum(entropy,dim=1)
                value,old_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=True)   # 最大熵的一半
                value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest = False)   # 最小熵的一半
                print(old_indices.shape,novel_indices.shape)
                pair_matrix = torch.mm(rela_center_vec,rela_center_vec.t())   # 构建了一致性矩阵，其实生成的伪配对

                pairs,pair_labels,_ = pairwise_sample(F.normalize(pred, 2, 1), batch_labels)   # 这里是真实的配对

                if args.use_cuda:
                    pairs.cuda()
                    pair_labels.cuda()
                    pair_matrix.cuda()
                    # initial_pair_matrix.cuda()
                    model.cuda()

                neg_values, novel_neg_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=False)
                pos_values, novel_pos_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=True)
                neg_values, old_neg_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=False)
                pos_values, old_pos_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=True)

                old_row = torch.LongTensor([[i] * args.oldnum for i in old_indices])
                old_row = old_row.reshape(-1).cuda() if args.use_cuda else old_row.reshape(-1)
                novel_row = torch.LongTensor([[i] * args.novelnum for i in novel_indices])
                novel_row = novel_row.reshape(-1).cuda() if args.use_cuda else novel_row.reshape(-1)
                row = torch.cat([old_row,novel_row])
                neg_ind = torch.cat([old_neg_ind.reshape(-1),novel_neg_ind.reshape(-1)])
                pos_ind = torch.cat([old_pos_ind.reshape(-1),novel_pos_ind.reshape(-1)])
                neg_distances = (pred[row] - pred[neg_ind]).pow(2).sum(1).unsqueeze(-1)
                pos_distances = (pred[row] - pred[pos_ind]).pow(2).sum(1).unsqueeze(-1)

                loss = torch.mean(torch.clamp(pos_distances + args.a - neg_distances, min=0.0))


                losses.append(loss.item())
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_train_batches.append(batch_seconds_spent)
                # end one batch

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.finetune_epochs, total_loss)
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_train_epochs.append(mins_spent)

            extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                  save_path_i, args, True)



        # Save model
        model_path = save_path_i + '/models'
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        p = model_path + '/finetune.pt'
        torch.save(model.state_dict(), p)
        print('finetune model saved after epoch ', str(epoch))

        # Save time spent on epochs
        np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
        print('Saved mins_train_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
        print('Saved seconds_train_batches.')

        return model


def graph_prompt_train(i, data_split, metrics, embedding_save_path, loss_fn, pretrain_model, label_center_emb, args, class_emb=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
    extract_features, extract_labels = extract_embeddings(g, pretrain_model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                            save_path_i, args, True)

    if args.prompt == "SUM":
        model = node_prompt_layer_sum()
    elif args.prompt == "LINEAR-MEAN":
        model = node_prompt_layer_linear_mean(args.gcn_hidden_dim * args.gcn_graph_num_layers,args.prompt_output_dim)
    elif args.prompt == "LINEAR-SUM":
        model = node_prompt_layer_linear_sum(args.gcn_hidden_dim * args.gcn_graph_num_layers, args.prompt_output_dim)
    elif args.prompt == "FEATURE-WEIGHTED-SUM":
        model = node_prompt_layer_feature_weighted_sum(args.gcn_hidden_dim * args.gcn_graph_num_layers)
    elif args.prompt == "FEATURE-WEIGHTED-MEAN":
        model = node_prompt_layer_feature_weighted_mean(args.gcn_hidden_dim * args.gcn_graph_num_layers)
    else:
        model = node_prompt_layer_sum()
        
    # model = model.to(device)
    model_param_group = []
    model_param_group.append({"params": model.parameters()})
    optimizer = torch.optim.AdamW(model_param_group, lr=args.lr,
                                    weight_decay=args.weight_decay, amsgrad=True)
        
    # Start prompt tuning
    message = "\n------------ Start prompt tuning------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.prompt_epochs):
        start_epoch = time.time()
        total_acc_socre = 0
        total_bp_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            model.train()
            start_batch = time.time()
            label_center_emb.to(device)

            trainemb = pretrain_model(blocks)
            pred = model(trainemb, 0)
            center_emb = model(label_center_emb, 0)

            # 预训练Loss函数
            # loss_outputs = loss_fn(pred, batch_labels)
            # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            # 比较和类别向量的相似性的Loss函数
            # loss, accfold = calc_loss(pred, device, args, batch_labels)

            # 伪标签的Loss函数
            pred = F.normalize(pred, 2, 1)
            rela_center_vec = torch.mm(pred,center_emb.t())
            rela_center_vec = F.normalize(rela_center_vec,2,1)
            epsilon = 1e-10
            rela_center_vec = torch.clamp(rela_center_vec, min=epsilon)
            entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            entropy = torch.sum(entropy,dim=1)
            value,old_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=True)   # 最大熵的一半
            value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest = False)   # 最小熵的一半

            # 开始构建一个样本
            total_size = len(old_indices) + len(novel_indices)

            # 创建一个全 0 的张量，大小为并集的大小
            combined_tensor = torch.zeros(total_size, dtype=torch.long)

            # 将 old_indices 对应的位置设置为 0（已经是 0，这一步可以省略）
            combined_tensor[old_indices] = 1

            # 将 novel_indices 对应的位置设置为 1
            combined_tensor[novel_indices] = 2

            # loss, accfold = calc_loss(pred, device, args, combined_tensor)
            loss,_ = prompt_loss(pred, combined_tensor, class_emb)

            # print(old_indices.shape,novel_indices.shape)
            # pair_matrix = torch.mm(rela_center_vec,rela_center_vec.t())   # 构建了一致性矩阵，其实生成的伪配对

            # pairs,pair_labels,_ = pairwise_sample(F.normalize(pred, 2, 1), batch_labels)   # 这里是真实的配对

            # if args.use_cuda:
            #     pairs.cuda()
            #     pair_labels.cuda()
            #     pair_matrix.cuda()
            #     # initial_pair_matrix.cuda()
            #     model.cuda()

            # neg_values, novel_neg_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=False)
            # pos_values, novel_pos_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=True)
            # neg_values, old_neg_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=False)
            # pos_values, old_pos_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=True)

            # old_row = torch.LongTensor([[i] * args.oldnum for i in old_indices])
            # old_row = old_row.reshape(-1).cuda() if args.use_cuda else old_row.reshape(-1)
            # novel_row = torch.LongTensor([[i] * args.novelnum for i in novel_indices])
            # novel_row = novel_row.reshape(-1).cuda() if args.use_cuda else novel_row.reshape(-1)
            # row = torch.cat([old_row,novel_row])
            # neg_ind = torch.cat([old_neg_ind.reshape(-1),novel_neg_ind.reshape(-1)])
            # pos_ind = torch.cat([old_pos_ind.reshape(-1),novel_pos_ind.reshape(-1)])
            # neg_distances = (pred[row] - pred[neg_ind]).pow(2).sum(1).unsqueeze(-1)
            # pos_distances = (pred[row] - pred[pos_ind]).pow(2).sum(1).unsqueeze(-1)

            # loss = torch.mean(torch.clamp(pos_distances + args.a - neg_distances, min=0.0))

            total_acc_socre += loss
            total_bp_loss += loss

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)

        total_acc_socre /= (batch_id + 1)
        total_bp_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average acc_score: {:.4f}. Average bp_loss: {:.4f}'.format(epoch + 1, args.prompt_epochs, total_acc_socre, total_bp_loss)
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, pretrain_model, len(labels), args, "graph_prompt", model)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                save_path_i, args, True)


    # Save model
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    p = model_path + '/finetune.pt'
    torch.save(model.state_dict(), p)
    print('finetune model saved after epoch ', str(epoch))

    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')


def gpf_prompt_train(i, data_split, metrics, embedding_save_path, loss_fn, pretrain_model, label_center_emb, args):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
    extract_features, extract_labels = extract_embeddings(g, pretrain_model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                            save_path_i, args, True)

    if args.prompt_type == 'gpf':
        model = SimplePrompt(args.out_dim)
    elif args.prompt_type == 'gpf-plus':
        model = GPFplusAtt(args.out_dim, args.pnum)

    model_param_group = []
    model_param_group.append({"params": model.parameters()})
    # if args.graph_pooling == "attention":
    #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})

    # Optimizer
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=1e-4, amsgrad=False)

    # Start prompt tuning
    message = "\n------------ Start prompt tuning------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.prompt_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            start_batch = time.time()
            # model.train()
            model.train()
            label_center_emb.to(device)

            # forward
            blocks[0].srcdata['features'] = model.add(blocks[0].srcdata['features'])
            pred = pretrain_model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow) add prompt vector

            # 伪标签的Loss函数
            pred = F.normalize(pred, 2, 1)
            rela_center_vec = torch.mm(pred,label_center_emb.t())
            rela_center_vec = F.normalize(rela_center_vec,2,1)
            epsilon = 1e-10
            rela_center_vec = torch.clamp(rela_center_vec, min=epsilon)
            entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            entropy = torch.sum(entropy,dim=1)
            value,old_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=True)   # 最大熵的一半
            value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest = False)   # 最小熵的一半

            # 开始构建一个样本
            total_size = len(old_indices) + len(novel_indices)

            # 创建一个全 0 的张量，大小为并集的大小
            combined_tensor = torch.zeros(total_size, dtype=torch.long)

            # 将 old_indices 对应的位置设置为 0（已经是 0，这一步可以省略）
            combined_tensor[old_indices] = 1

            # 将 novel_indices 对应的位置设置为 1
            combined_tensor[novel_indices] = 2

            loss,_ = prompt_loss(pred, combined_tensor)

            

            # 预训练Loss
            # loss_outputs = loss_fn(pred, batch_labels)
            # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs


            # pred = F.normalize(pred, 2, 1)
            # rela_center_vec = torch.mm(pred,label_center_emb.t())
            # rela_center_vec = F.normalize(rela_center_vec,2,1)
            # entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            # entropy = torch.sum(entropy,dim=1)
            # value,old_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=True)   # 最大熵的一半
            # value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest = False)   # 最小熵的一半
            # print(old_indices.shape,novel_indices.shape)
            # pair_matrix = torch.mm(rela_center_vec,rela_center_vec.t())   # 构建了一致性矩阵，其实生成的伪配对

            # pairs,pair_labels,_ = pairwise_sample(F.normalize(pred, 2, 1), batch_labels)   # 这里是真实的配对

            # if args.use_cuda:
            #     pairs.cuda()
            #     pair_labels.cuda()
            #     pair_matrix.cuda()
            #     # initial_pair_matrix.cuda()
            #     model.cuda()

            # neg_values, novel_neg_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=False)
            # pos_values, novel_pos_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=True)
            # neg_values, old_neg_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=False)
            # pos_values, old_pos_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=True)

            # old_row = torch.LongTensor([[i] * args.oldnum for i in old_indices])
            # old_row = old_row.reshape(-1).cuda() if args.use_cuda else old_row.reshape(-1)
            # novel_row = torch.LongTensor([[i] * args.novelnum for i in novel_indices])
            # novel_row = novel_row.reshape(-1).cuda() if args.use_cuda else novel_row.reshape(-1)
            # row = torch.cat([old_row,novel_row])
            # neg_ind = torch.cat([old_neg_ind.reshape(-1),novel_neg_ind.reshape(-1)])
            # pos_ind = torch.cat([old_pos_ind.reshape(-1),novel_pos_ind.reshape(-1)])
            # neg_distances = (pred[row] - pred[neg_ind]).pow(2).sum(1).unsqueeze(-1)
            # pos_distances = (pred[row] - pred[pos_ind]).pow(2).sum(1).unsqueeze(-1)

            # loss = torch.mean(torch.clamp(pos_distances + args.a - neg_distances, min=0.0))


            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.prompt_epochs, total_loss)
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        # with open(save_path_i + '/log.txt', 'a') as f:
        #     f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, pretrain_model, len(labels), args, "gpf", model)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                save_path_i, args, True)


    # Save model
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    p = model_path + '/finetune.pt'
    torch.save(pretrain_model.state_dict(), p)
    print('finetune model saved after epoch ', str(epoch))

    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')

    return pretrain_model

def graphpro_prompt_train(i, data_split, metrics, embedding_save_path, model, label_center, args, old_label_rate, score_result=[]):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices, adjacency_matrix = getdata(
        embedding_save_path, data_split, i, args)

    # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                            save_path_i, args, True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, amsgrad=False)

    # Start prompt tuning
    message = "\n------------ Start prompt tuning------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.prompt_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            start_batch = time.time()
            model.train()

            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow) add prompt vector

            # 伪标签的Loss函数
            # pred_norm = F.normalize(pred, 2, 1)
            rela_center_vec = torch.mm(pred,label_center_emb.t())
            rela_center_vec = F.normalize(rela_center_vec,2,1)
            epsilon = 1e-10
            rela_center_vec = torch.clamp(rela_center_vec, min=epsilon)
            entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            entropy = torch.sum(entropy,dim=1)
            value,old_indices = torch.topk(entropy.reshape(-1),int(old_label_rate*entropy.shape[0]),largest=True)   # 最大熵的一半
            # value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=False)   # 最小熵的一半

            # 模型预测结果，通过欧式距离判断属于已知类的那类
            distance = distance2center2(pred, label_center_emb)
            distance = 1/F.normalize(distance, dim=1)
            label_pred = F.log_softmax(distance, dim=1)
            label_pred = torch.argmax(label_pred, dim=1, keepdim=True).squeeze()  # 这里是判断所有预测node，属于label_center_emb的那个类别

            # 开始构建一个样本，创建一个新类的标识数组
            pseudo_new_label = len(label_center.keys())
            combined_tensor = torch.full((pred.shape[0],), pseudo_new_label)
            combined_tensor[old_indices] = label_pred[old_indices]
            loss,_pred = prompt_loss(pred, combined_tensor, adjacency_matrix)
            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证 pseudo old label 是否准确
            pseudo_old_labels = batch_labels[old_indices]
            print('pseudo old event:',len(set(pseudo_old_labels.tolist())))
            print('true old event:',len(label_center.keys()))
            print(set(label_center.keys()) & set(pseudo_old_labels.tolist()))

            # 构建一个 pseudo old&new label
            true_labels = process_tensor(batch_labels, label_center.keys())
            current_old_label_rate = torch.sum(true_labels == 0).item() / true_labels.shape[0]  # 计算当前旧类占比
            true_labels[true_labels == 1] = pseudo_new_label
            true_labels[old_indices] = label_pred[old_indices] 

            for metric in metrics:
                metric(_pred, true_labels, loss)   # combined_tensor

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.prompt_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        # with open(save_path_i + '/log.txt', 'a') as f:
        #     f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        try:
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                    save_path_i, args, True)
        except Exception as e:
            print(e)
        # 比较分数
        if test_value >= MAX_SCORE[str(i)]:
            score_result.append(f' Epoch {epoch+1} : {test_value}')

    # Save model
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    # p = model_path + '/finetune.pt'
    # torch.save(model.state_dict(), p)
    # print('finetune model saved after epoch ', str(epoch))

    # update & save label_center
    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    for l in set(extract_labels):
        l_indices = np.where(extract_labels==l)[0]
        l_feas = extract_features[l_indices]

        # 通过判断历史的计算值，可以构建一个移动平均，使得平均值更加稳定
        if l in label_center:
            label_center_expanded = np.expand_dims(label_center[l], axis=0)
            l_feas = np.concatenate((l_feas, label_center_expanded))

        l_cen = np.mean(l_feas,0)
        label_center[l] = l_cen
    # joblib.dump(label_center,save_path_i + '/models/label_center.dump')

    # 对旧类占比求平均
    old_label_rate = (old_label_rate + current_old_label_rate) / 2 
    print(old_label_rate)
    # # Save time spent on epochs
    # np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    # print('Saved mins_train_epochs.')
    # # Save time spent on batches
    # np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    # print('Saved seconds_train_batches.')

    return model,label_center,old_label_rate

def graphpro_prompt_train_new(i, data_split, metrics, embedding_save_path, model, label_center, args, old_label_rate, score_result=[]):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices, adjacency_matrix = getdata(
        embedding_save_path, data_split, i, args)

    # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                            save_path_i, args, True)
    
    p_num = len(set(labels.tolist()))
    n_prompt = GPFplusAtt(in_feats, p_num)   # args.out_dim
    # n_prompt = GPFplusAtt(args.out_dim, p_num)
    c_prompt = CenterEmbedding(args.out_dim, p_num)

    model_param_group = []
    model_param_group.append({"params": n_prompt.parameters()})
    model_param_group.append({"params": c_prompt.parameters()})

    # Optimizer
    optimizer = optim.Adam(params=model_param_group, lr=args.lr, weight_decay=1e-4, amsgrad=False)

    # Start prompt tuning
    message = "\n------------ Start prompt tuning------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.prompt_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            start_batch = time.time()
            n_prompt.train()
            c_prompt.train()

            # forward
            blocks[0].srcdata['features'] = n_prompt.add(blocks[0].srcdata['features'])
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow) add prompt vector
            # pred = n_prompt.add(pred)

            # 伪标签的Loss函数
            # pred_norm = F.normalize(pred, 2, 1)
            rela_center_vec = torch.mm(pred,label_center_emb.t())
            rela_center_vec = F.normalize(rela_center_vec,2,1)
            epsilon = 1e-10
            rela_center_vec = torch.clamp(rela_center_vec, min=epsilon)
            entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            entropy = torch.sum(entropy,dim=1)
            value,old_indices = torch.topk(entropy.reshape(-1),int(old_label_rate*entropy.shape[0]),largest=True)   # 最大熵的一半
            # value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=False)   # 最小熵的一半

            # 模型预测结果，通过欧式距离判断属于已知类的那类
            distance = distance2center2(pred, label_center_emb)
            distance = 1/F.normalize(distance, dim=1)
            label_pred = F.log_softmax(distance, dim=1)
            label_pred = torch.argmax(label_pred, dim=1, keepdim=True).squeeze()  # 这里是判断所有预测node，属于label_center_emb的那个类别

            # 开始构建一个样本，创建一个新类的标识数组
            pseudo_new_label = len(label_center.keys())
            pseudo_labels = torch.full((pred.shape[0],), pseudo_new_label)
            pseudo_labels[old_indices] = label_pred[old_indices]
            c_embedding_prompt = c_prompt(pred, pseudo_labels)
            loss,_pred = prompt_loss_with_center(pred, pseudo_labels, c_embedding_prompt, adjacency_matrix)
            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证 pseudo old label 是否准确
            pseudo_old_labels = batch_labels[old_indices]
            print('pseudo old event:',len(set(pseudo_old_labels.tolist())))
            print('true old event:',len(label_center.keys()))
            print(set(label_center.keys()) & set(pseudo_old_labels.tolist()))

            # 构建一个 pseudo old&new label
            true_labels = process_tensor(batch_labels, label_center.keys())
            current_old_label_rate = torch.sum(true_labels == 0).item() / true_labels.shape[0]  # 计算当前旧类占比
            true_labels[true_labels == 1] = pseudo_new_label
            true_labels[old_indices] = label_pred[old_indices] 

            for metric in metrics:
                metric(_pred, true_labels, loss)   # combined_tensor

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.prompt_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        # with open(save_path_i + '/log.txt', 'a') as f:
        #     f.write(message)
        mins_train_epochs.append(mins_spent)

        # extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, prompt='prompt', prompt_model=n_prompt)

        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                save_path_i, args, True)

        # 比较分数
        if test_value >= MAX_SCORE[str(i)]:
            score_result.append(f' Epoch {epoch+1} : {test_value}')

    # Save model
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    # p = model_path + '/finetune.pt'
    # torch.save(model.state_dict(), p)
    # print('finetune model saved after epoch ', str(epoch))

    with torch.no_grad():
        c_prompt.eval()
        c_embedding_prompt = c_prompt(torch.FloatTensor(extract_features), torch.LongTensor(extract_labels))

    # update & save label_center
    for l in set(extract_labels):
        # l_indices = np.where(extract_labels==l)[0]
        # l_feas = extract_features[l_indices]

        # # 通过判断历史的计算值，可以构建一个移动平均，使得平均值更加稳定
        # if l in label_center:
        #     label_center_expanded = np.expand_dims(label_center[l], axis=0)
        #     l_feas = np.concatenate((l_feas, label_center_expanded))

        # l_cen = np.mean(l_feas,0)
        # label_center[l] = l_cen

        # 使用CenterEmbedding
        label_center[l] = c_embedding_prompt[l]
    # joblib.dump(label_center,save_path_i + '/models/label_center.dump')

    # 对旧类占比求平均
    old_label_rate = (old_label_rate + current_old_label_rate) / 2 
    print(old_label_rate)
    # # Save time spent on epochs
    # np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    # print('Saved mins_train_epochs.')
    # # Save time spent on batches
    # np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    # print('Saved seconds_train_batches.')

    return model,label_center,old_label_rate

def prompt_domain_train(i, data_split, metrics, embedding_save_path, loss_fn, model, clip_model, label_center_emb, args):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                            save_path_i, args, True)
    
    # EMA update
    def update_ema_buffer(da_head, alpha, iteration):
        alpha = min(1 - 1 / (iteration + 1), alpha)

        da_head.prompt_learner.ctx_di_ema.mul_(alpha).add_(1 - alpha, da_head.prompt_learner.ctx_di)
        da_head.prompt_learner.ctx_ds_ema.mul_(alpha).add_(1 - alpha, da_head.prompt_learner.ctx_ds)


    # learnable prompt embeddings
    for params in clip_model.parameters():
        params.requires_grad_(False)
    DAHead = None # DAPromptHead(args.prompt_class, clip_model, args.ctx_size)

    
    
     
    if args.prompt_type == 'gpf':
        prompt = SimplePrompt(args.out_dim)
    elif args.prompt_type == 'gpf-plus':
        prompt = GPFplusAtt(args.out_dim, args.pnum)

    model_param_group = []
    model_param_group.append({"params": prompt.parameters()})
    # if args.graph_pooling == "attention":
    #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})

    # Optimizer
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=1e-4, amsgrad=False)

    # Start prompt tuning
    message = "\n------------ Start prompt tuning------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.prompt_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            start_batch = time.time()
            # model.train()
            prompt.train()
            label_center_emb.to(device)

            # forward
            blocks[0].srcdata['features'] = prompt.add(blocks[0].srcdata['features'])
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow) add prompt vector

            normalized_x = F.normalize(pred, p=2.0, dim=1)

            # learnable prompt embeddings
            text_embedding = DAHead.get_embedding() #[domains * (cls), 1024]

            text_embedding = F.normalize(text_embedding, p=2.0, dim=1)
            da_cls_scores = normalized_x @ text_embedding.t()
            da_cls_scores_source = da_cls_scores[:, :args.num_classes]
            da_cls_scores_target = da_cls_scores[:, args.num_classes:]

            # EMA embeddings
            text_embedding_ema = DAHead.get_embedding_ema().detach() #[domains * (cls), 1024]

            text_embedding_ema = F.normalize(text_embedding_ema, p=2.0, dim=1)
            ema_cls_scores = normalized_x @ text_embedding_ema.t()
            ema_cls_scores_source = ema_cls_scores[:, :args.num_classes]
            ema_cls_scores_target = ema_cls_scores[:, args.num_classes:]

            da_scores = torch.cat((da_cls_scores_source, da_cls_scores_target), dim=1)   
            da_scores = da_scores / args.temperature 

            #EMA scores
            ema_scores = torch.cat((ema_cls_scores_source, ema_cls_scores_target), dim=1)   
            ema_scores = ema_scores / args.temperature
            
            # if prompt tuning, use origin scores as pseudo labels, and use da_scores as logits
            # 这里涉及构建pseudo和得分
            pseudo_scores = None
            scores = da_scores

            #####################################
            N, D_C = scores.shape
            C = int(D_C / 2)
            score_across_domains = scores
            score_source = scores[:, :C]
            score_target = scores[:, C:]
            if args.is_source:
                scores = score_source
            else:
                scores = score_target

            # train learnable prompt embeddings
            if args.is_source:
                losses["loss_across_domains"] = focal_loss(score_across_domains, batch_labels)
                losses["loss_target_domain"] = focal_loss(score_target, batch_labels)
                # source domain do not need teacher
            # pseudo loss
            if not args.is_source:
                pseudo_scores = pseudo_scores[:, C:] 
                pseudo_label = torch.softmax(pseudo_scores, dim=-1).detach()
                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(0.5).float()
                C_label_p = label_p + C
                losses['loss_pseudo_target_domain'] = (F.cross_entropy(
                        score_target, label_p, reduction="none") * mask).sum() / mask.sum()
                losses['loss_target_entropy'] = - (pseudo_label * torch.log_softmax(score_target, dim=-1)).sum() / N
                losses['loss_pseudo_across_domain'] = 0.25 * (F.cross_entropy(
                        score_across_domains, C_label_p, reduction="none") * mask).sum() / mask.sum()


            loss_outputs = loss_fn(pred, batch_labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs


            # pred = F.normalize(pred, 2, 1)
            # rela_center_vec = torch.mm(pred,label_center_emb.t())
            # rela_center_vec = F.normalize(rela_center_vec,2,1)
            # entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            # entropy = torch.sum(entropy,dim=1)
            # value,old_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=True)   # 最大熵的一半
            # value,novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest = False)   # 最小熵的一半
            # print(old_indices.shape,novel_indices.shape)
            # pair_matrix = torch.mm(rela_center_vec,rela_center_vec.t())   # 构建了一致性矩阵，其实生成的伪配对

            # pairs,pair_labels,_ = pairwise_sample(F.normalize(pred, 2, 1), batch_labels)   # 这里是真实的配对

            # if args.use_cuda:
            #     pairs.cuda()
            #     pair_labels.cuda()
            #     pair_matrix.cuda()
            #     # initial_pair_matrix.cuda()
            #     model.cuda()

            # neg_values, novel_neg_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=False)
            # pos_values, novel_pos_ind = torch.topk(pair_matrix[novel_indices], args.novelnum, 1, largest=True)
            # neg_values, old_neg_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=False)
            # pos_values, old_pos_ind = torch.topk(pair_matrix[old_indices], args.oldnum, 1, largest=True)

            # old_row = torch.LongTensor([[i] * args.oldnum for i in old_indices])
            # old_row = old_row.reshape(-1).cuda() if args.use_cuda else old_row.reshape(-1)
            # novel_row = torch.LongTensor([[i] * args.novelnum for i in novel_indices])
            # novel_row = novel_row.reshape(-1).cuda() if args.use_cuda else novel_row.reshape(-1)
            # row = torch.cat([old_row,novel_row])
            # neg_ind = torch.cat([old_neg_ind.reshape(-1),novel_neg_ind.reshape(-1)])
            # pos_ind = torch.cat([old_pos_ind.reshape(-1),novel_pos_ind.reshape(-1)])
            # neg_distances = (pred[row] - pred[neg_ind]).pow(2).sum(1).unsqueeze(-1)
            # pos_distances = (pred[row] - pred[pos_ind]).pow(2).sum(1).unsqueeze(-1)

            # loss = torch.mean(torch.clamp(pos_distances + args.a - neg_distances, min=0.0))


            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            update_ema_buffer(DAHead, 0.99, args.iter)

            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.prompt_epochs, total_loss)
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        # with open(save_path_i + '/log.txt', 'a') as f:
        #     f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args, prompt)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                save_path_i, args, True)


    # Save model
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    p = model_path + '/finetune.pt'
    torch.save(model.state_dict(), p)
    print('finetune model saved after epoch ', str(epoch))

    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')

    return model

def graphpro_fine_tune_train(i, data_split, embedding_save_path, model, label_center, args):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices, adjacency_matrix = getdata(
        embedding_save_path, data_split, i, args)

    # 训练之前先搞下评估，然后看看一个block完成后，评估的比较
    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    test_value = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                            save_path_i, args, True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, amsgrad=False)

    # Start prompt tuning
    message = "\n------------ Start prompt tuning------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda() if args.use_cuda else torch.FloatTensor(np.array(list(label_center.values())))
    
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.prompt_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, test_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]

            start_batch = time.time()
            model.train()

            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow) add prompt vector

            # 伪标签的Loss函数
            pred_norm = F.normalize(pred, 2, 1)
            rela_center_vec = torch.mm(pred_norm,label_center_emb.t())
            rela_center_vec = F.normalize(rela_center_vec,2,1)
            epsilon = 1e-10
            rela_center_vec = torch.clamp(rela_center_vec, min=epsilon)
            entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
            entropy = torch.sum(entropy,dim=1)
            value, old_indices = torch.topk(entropy.reshape(-1),int(args.topk),largest=True)   # 最大熵的一半
            # value, novel_indices = torch.topk(entropy.reshape(-1),int(entropy.shape[0]/2),largest=False)   # 最小熵的一半

            # 已知类预测的
            distance = distance2center2(pred, label_center_emb)
            distance = 1/F.normalize(distance, dim=1)
            label_pred = F.log_softmax(distance, dim=1)
            label_pred = torch.argmax(label_pred, dim=1, keepdim=True).squeeze()

            # 开始构建一个样本，创建一个新类的标识数组
            combined_tensor = torch.full((pred.shape[0],), max(label_center.keys()) + 1)
            combined_tensor[old_indices] = label_pred[old_indices]

            loss, _pred = prompt_loss(pred, combined_tensor, adjacency_matrix)

            losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.prompt_epochs, total_loss)
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        # with open(save_path_i + '/log.txt', 'a') as f:
        #     f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)

        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes,
                                save_path_i, args, True)

    # Save model
    model_path = save_path_i + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    p = model_path + '/finetune.pt'
    torch.save(model.state_dict(), p)
    print('finetune model saved after epoch ', str(epoch))

    # update & save label_center
    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    for l in set(extract_labels):
        l_indices = np.where(extract_labels==l)[0]
        l_feas = extract_features[l_indices]

        # 通过判断历史的计算值，可以构建一个移动平均，使得平均值更加稳定
        if l in label_center:
            label_center_expanded = np.expand_dims(label_center[l], axis=0)
            l_feas = np.concatenate((l_feas, label_center_expanded))

        l_cen = np.mean(l_feas,0)
        label_center[l] = l_cen
    joblib.dump(label_center,save_path_i + '/models/label_center.dump')

    return model,label_center,test_value
