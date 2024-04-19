import joblib
from train import graph_prompt_train, graphpro_prompt_train, initial_train,continue_train,gpf_prompt_train, initial_train_with_prompt_loss
import argparse
from utils import OnlineTripletLoss
from utils import HardestNegativeTripletSelector
from utils import RandomNegativeTripletSelector
from metric import AccuracyMetric, AverageNonzeroTripletsMetric, MacroF1Metric
import torch
from time import localtime, strftime
import os
import json
import numpy as np
from model_dynamic import GAT
# from model import GAT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper parameters
    parser.add_argument('--finetune_epochs', default=3, type=int, #embeddings_0430063028
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--prompt_epochs', default=15, type=int,
                        help="Number of prompt tuning epochs.")
    parser.add_argument('--n_epochs', default=15, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--oldnum', default=20, type=int,
                        help="Number of sampling.")
    parser.add_argument('--novelnum', default=10, type=int,
                        help="Number of sampling.")
    parser.add_argument('--n_infer_epochs', default=0, type=int,
                        help="Number of inference epochs.")
    parser.add_argument('--window_size', default=3, type=int,
                        help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--margin', default=3., type=float,
                        help="Margin for computing triplet losses")
    parser.add_argument('--a', default=16., type=float,
                        help="Margin for computing pair-wise losses")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--batch_size', default=2100, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.")
    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', default=16, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=64, type=int,
                        help="Output dimension of tweet representations")   # 64
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads in each GAT layer")
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")
    parser.add_argument('--prompt_type', dest='prompt_type', default='gpf',
                        help="Choise prompt type")
    parser.add_argument('--validation_percent', default=0.1, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--test_percent', default=0.2, type=float,
                        help="Percentage of test nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False,
                        action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--metrics', type=str, default='nmi')
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--add_ort', dest='add_ort', default=True,
                        action='store_true',
                        help="Use orthorgonal constraint")
    parser.add_argument('--gpuid', type=int, default=3)
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")
    # offline or online situation
    parser.add_argument('--is_incremental', default=True, action='store_true')
    parser.add_argument('--data_path', default='./data/0413_ALL_English',   # 0413_ALL_English  0413_ALL_French  incremental_continue_graph  incremental_continue_graph_french
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--add_pair', action='store_true', default=True)

    # 使用GraphPrompt框架的参数
    parser.add_argument('--prompt', default='LINEAR-SUM')  # FEATURE-WEIGHTED-SUM   LINEAR-SUM
    parser.add_argument('--gcn_hidden_dim', default=300)
    parser.add_argument('--gcn_graph_num_layers', default=1)
    parser.add_argument('--prompt_output_dim', default=300)   # 64
    parser.add_argument('--weight_decay', default=0.00001)
    parser.add_argument('--reg_loss', default='MSE')
    parser.add_argument('--bp_loss', default='MSE')
    parser.add_argument('--bp_loss_slp',default='anneal_cosine$1.0$0.01')

    args = parser.parse_args()
    use_cuda = False  # True
    print("Using CUDA:", use_cuda)
    if use_cuda:
        torch.cuda.set_device(args.gpuid)

    embedding_save_path = args.data_path + '/embeddings_0222164008_graphprompt' # + strftime("%m%d%H%M%S", localtime())   embeddings_continue_graphprompt embeddings_0222164008_graphprompt
    # os.mkdir(embedding_save_path)
    print("embedding_save_path: ", embedding_save_path)
    with open(embedding_save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # [500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    data_split = np.load(args.data_path + '/data_split.npy')

    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

    # Metrics
    metrics = [AccuracyMetric(),MacroF1Metric()]
    # class_emb = torch.load("./data/0413_ALL_English/class_features.pth")
    # pre_train_labels = set(np.load("./data/0413_ALL_English/0/labels.npy"))
    
    if args.add_pair:
        # model, label_center_emb = initial_train(0, args, data_split, metrics, embedding_save_path, loss_fn, None)
        model = GAT(302, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual, 'finetune')   #, 'finetune'
        best_model_path = f"{embedding_save_path}/block_0/models/best.pt"   # 基于initial_train的最好模型开始进行下面的微调
        # label_center_emb = torch.load(f"{embedding_save_path}/block_0/models/center.pth")
        label_center = joblib.load(f"{embedding_save_path}/block_0/models/label_center.dump")

        # 加载预训练的模型参数，在此基础上添加，初始化的GateLayer参数
        state_dict = torch.load(best_model_path)
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if any(k.startswith(s) for s in ["user_embedding", "item_embedding"]):
        #         new_state_dict[k] = v

        model.load_state_dict(state_dict, strict=False)
        if args.use_cuda:
            model.cuda()

        if args.is_incremental:
            message = ""

            # 从1开始，因为0是initial_train，从1~21个block，这里可以从0开始，修改continue_train函数中的代码即可，在为0时进行一次backbone的评估
            for i in range(1, data_split.shape[0]):
                print("incremental setting")
                print("enter i ",str(i))
                max_score = []

                # Inference (prediction)
                # model = continue_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center_emb, args)
                # graph_prompt_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center_emb, args, None )   # None  
                # gpf_prompt_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center_emb, args)
                model, label_center = graphpro_prompt_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center, args, score_result=max_score)   # class_emb
                message += f"\n M{i}: {max_score}" 

            # 输出最好结果
            print(message)

    else:
        # pre-training
        model = initial_train(0, args, data_split, metrics, embedding_save_path, loss_fn, None)
        # model = initial_train_with_prompt_loss(0, args, data_split, metrics, embedding_save_path, class_emb, None)
