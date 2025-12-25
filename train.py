import argparse
import os
import random
import time
import uuid

import numpy as np
import torch

import utils.dataUtils as du
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold

from model import LMHGL, HeatDiffusion
from utils.LoggerFactory import get_logger
from sklearn.metrics import accuracy_score
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


def set_env(seed):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_env(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    checkpt_file = os.getcwd() + '/pretrained/' + uuid.uuid4().hex + '.pt'

    data_path = os.getcwd() + '/dataset/' + args.dataset
    adj, features, labels = du.load_dataset(data_path, balence_weights=False, normalized='sys', task=args.task)

    R_adj, R_features, L_adj, L_features = du.divide_brain_network_L_and_R(adj, features)
    ucn_adj,ucn_features,ucn_nodes = du.get_ucn_brain_network(adj, features)

    R_adj = torch.from_numpy(R_adj).to(device)
    R_features = torch.from_numpy(R_features).to(device)
    L_adj = torch.from_numpy(L_adj).to(device)
    L_features = torch.from_numpy(L_features).to(device)
    ucn_features = torch.from_numpy(ucn_features).to(device)
    ucn_adj = torch.from_numpy(ucn_adj).to(device)

    t_total = time.time()
    diffusion = HeatDiffusion(t=args.t, top_k=args.top_k, threshold=args.threshold)
    R_adj = diffusion.process(R_adj)
    L_adj = diffusion.process(L_adj)

    Left_brain_graph = du.create_data_list(L_features,L_adj,'left')
    Right_brain_graph = du.create_data_list(R_features,R_adj,'right')
    UCN_brain_graph = du.create_data_list(ucn_features,ucn_adj,'ucn')

    metrics = du.Metrics()

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True,random_state=args.seed)

    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        model = LMHGL(input_dim=features.shape[2],
                      hidden_dim=args.hidden,
                      out_dim=int(labels.max()) + 1,
                      num_layers=args.layers,
                      alpha=args.alpha,
                      theta=args.lamda,
                      dropout=args.dropout,
                      lamba=0.2,
                      enta=0.7,
                      device=device
                      ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 训练集索引
        left_train_graph = [Left_brain_graph[i] for i in train_index]
        left_test_graph = [Left_brain_graph[i] for i in test_index]
        right_train_graph = [Right_brain_graph[i] for i in train_index]
        right_test_graph = [Right_brain_graph[i] for i in test_index]
        ucn_train_graph = [UCN_brain_graph[i] for i in train_index]
        ucn_test_graph = [UCN_brain_graph[i] for i in test_index]

        left_train_graph = next(iter(DataLoader(left_train_graph, batch_size=args.batch_size, shuffle=False)))

        left_test_graph = next(iter(DataLoader(left_test_graph, batch_size=args.batch_size, shuffle=False)))
        right_train_graph = next(iter(DataLoader(right_train_graph, batch_size=args.batch_size, shuffle=False)))
        right_test_graph = next(iter(DataLoader(right_test_graph, batch_size=args.batch_size, shuffle=False)))
        ucn_train_graph = next(iter(DataLoader(ucn_train_graph, batch_size=args.batch_size, shuffle=False)))
        ucn_test_graph = next(iter(DataLoader(ucn_test_graph, batch_size=args.batch_size, shuffle=False)))

        best = 0
        count = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(left_train_graph,right_train_graph,ucn_train_graph)
            target = torch.from_numpy(labels[train_index]).long().to(device)
            loss = F.nll_loss(output, target)
            pred = torch.softmax(output, dim=1)
            pred = torch.max(pred, 1)[1].view(-1)
            loss.backward()
            optimizer.step()
            logger.info('Fold {}, Epoch {}, loss {:.4f}, acc {:.4f}'.format(fold, epoch, loss.to('cpu'), accuracy_score(labels[train_index], pred.to('cpu'))))

            model.eval()
            with torch.no_grad():
                output= model(left_test_graph,right_test_graph,ucn_test_graph)
                pred = torch.softmax(output, 1)
                pred = torch.max(pred, 1)[1].view(-1)

                acc_val = metrics.calculate_accuracy(labels[test_index], pred.to('cpu'))
                logger.warning('Epoch {}, acc {:.4f}'.format(epoch, acc_val))
                if best < acc_val:
                    best = acc_val
                    torch.save(model.state_dict(), checkpt_file)
                    count = 0
                else:
                    count += 1
                if count == args.patience:
                    break
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        with torch.no_grad():
            output = model(left_test_graph,right_test_graph,ucn_test_graph)
            pred = torch.softmax(output, 1)
            pred = torch.max(pred, 1)[1].view(-1)
            print(labels[test_index], pred.to('cpu'))
            metrics.calculate_all_metrics(labels[test_index], pred.to('cpu'), average='weighted')

    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logger.info('The accuracy is {}.'.format(metrics.accuracy_in_all_metrics()))
    logger.info('The list of model initialization parameters is {}'.format(args.__dict__))
    logger.info(
        'The average accuracy is {}%.'.format(metrics.accuracy_average_in_all_metrics()))
    metrics.average_all_metrics()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', type=int, default=1, help='GPU device ID to use (default: 0).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number layers: At least greater than 2 to take effect, GCN at least two layers.')
    parser.add_argument('--t', type=float, default=4,
                        help='Diffusion time.')
    parser.add_argument('--top_k', type=float, default=30,
                        help='Diffusion time.')
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='Diffusion time.')
    parser.add_argument('--kfold', type=int, default=5,
                        help='k-fold: Cross-validate the fold.')
    parser.add_argument('--batch_size', type=int, default=300,
                        help='batch_size: Each time the batch_size is read.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--dataset', type=str, default='/Storke/Storke.pkl',
                        help='Dataset string')
    parser.add_argument('--task', type=str,
                        default="{'subtype0': ['NC_A'], 'subtype1': ['SD']}",
                        help='k-fold: Cross-validate the fold.')
    args = parser.parse_args()

    logger = get_logger()
    logger.info(args)

    main(args)