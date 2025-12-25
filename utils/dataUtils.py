import numpy as np
import pickle
from torch_geometric.data import Data

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch_geometric.utils import dense_to_sparse

from utils.LoggerFactory import get_logger

logger = get_logger()

def is_has_nan(matrix: np.ndarray) -> bool:
    if np.isnan(matrix).any():
        return True
    else:
        return False

def to_sys_matrix(adj: np.ndarray) -> np.ndarray:
    adj = adj + adj.transpose(0, 2, 1) * (adj.transpose(0, 2, 1) > adj) - adj * (adj.transpose(0, 2, 1) > adj)
    return adj

def balance_weights_to_ones(matrix: np.array) -> np.ndarray:
    matrix[matrix != 0] = 1
    return matrix

def add_self_loops(adj: np.ndarray) -> np.ndarray:
    identity_matrix = np.eye(adj.shape[1])
    return adj + identity_matrix

def sys_normalized_adjacency(adj: np.ndarray, is_self_edge: bool = True) -> np.ndarray:
    """
    Symmetric normalization: D^{-1/2} * Adj * D^{-1/2}
    :param adj: adjacency matrix
    :return: symmetrically normalized adjacency matrix
    """
    if is_self_edge:
        adj = add_self_loops(adj)
    for idx in range(adj.shape[0]):
        row_sum = adj[idx].sum(axis=1)
        D_inv = np.power(row_sum, -0.5).flatten()
        D_inv[np.isinf(D_inv)] = 0.
        D_diag = np.diag(D_inv)
        adj[idx] = D_diag @ adj[idx] @ D_diag
    return adj


def rw_normalized_adjacency(adj, is_self_edge: bool = True):
    """
    Random walk normalized adjacency matrix: D^{-1} Adj  D^{-1}
    :param adj: adjacency matrix
    :return: random walk normalized adjacency matrix
    """
    if is_self_edge:
        adj = add_self_loops(adj)
    for idx in range(adj.shape[0]):
        row_sum = adj[idx].sum(axis=1)
        D_inv = np.power(row_sum, -1).flatten()
        D_inv[np.isinf(D_inv)] = 0.
        D_diag = np.diag(D_inv)
        adj[idx] = D_diag @ adj[idx]
    return adj

def average_pooling(matrix, target_rows):
    original_rows, cols = matrix.shape
    pooled_matrix = np.zeros((target_rows, cols))
    step = original_rows / target_rows
    for i in range(target_rows):
        start = int(i * step)
        end = min(int(np.ceil((i + 1) * step)), original_rows)
        pooled_matrix[i, :] = np.mean(matrix[start:end, :], axis=0)
    return pooled_matrix

# task: dict = {'subtype0': ['NC_A','NC_B'], 'subtype1': ['MCI']}
def load_pickle(path: str = './', task =  None):

    with open(path, 'rb') as file:
        # 加载数据
        data = pickle.load(file)
    dict_task = eval(task)
    sub_0 = dict_task['subtype0']
    fmri_matrix_0 = np.vstack([data[key][0] for key in sub_0])

    pooled_brain_networks = np.zeros((fmri_matrix_0.shape[0], 118, 90))
    #Apply average pooling function to each (187, 90) matrix
    for i in range(28):
        pooled_brain_networks[i] = average_pooling(fmri_matrix_0[i], 118)

    dti_matrix_0 = np.vstack([data[key][1] for key in sub_0])
    label_matrix_0 = np.concatenate([data[key][2] for key in sub_0])
    label_matrix_0[:] = 0

    sub_1 = dict_task['subtype1']
    fmri_matrix_1 = np.vstack([data[key][0] for key in sub_1])
    dti_matrix_1 = np.vstack([data[key][1] for key in sub_1])
    label_matrix_1 = np.concatenate([data[key][2] for key in sub_1])
    label_matrix_1[:] = 1

    fmri_matrix = np.vstack((pooled_brain_networks, fmri_matrix_1))
    dti_matrix = np.vstack((dti_matrix_0, dti_matrix_1))
    label_matrix = np.concatenate((label_matrix_0, label_matrix_1))

    assert fmri_matrix.shape[0] == dti_matrix.shape[0] == len(label_matrix)

    return dti_matrix, fmri_matrix, label_matrix

def load_dataset(dataset: str, balence_weights: bool = False, normalized: str = None, task = None):
    adj, features, labels = load_pickle(dataset, task)
    features = np.transpose(features, (0, 2, 1))

    if is_has_nan(labels) or is_has_nan(features) or is_has_nan(adj):
        raise ValueError("Encountered NaN value form labels or features or adj.")

    # adj[adj<0] = 0
    adj = to_sys_matrix(adj)

    if balence_weights:
        adj = balance_weights_to_ones(adj)
    if normalized == None:
        pass
    elif normalized.lower() == 'sys':
        adj = sys_normalized_adjacency(adj)
    elif normalized.lower() == 'rw':
        adj = rw_normalized_adjacency(adj)

    return adj.astype(np.float32), features.astype(np.float32), labels.astype(np.int32)


def get_ucn_brain_network(adj, features):
    node_bumbers = features.shape[1]
    masks,cross_and_neighbors = get_node_mask_batch(adj)
    ucn_adj = adj * masks
    for i in range(len(features)):
        mask = np.zeros(node_bumbers, dtype=bool)
        mask[cross_and_neighbors[i]] = True  # 要保留的节点标记为True
        features[i, ~mask, :] = 0  # 关键修正：添加冒号处理所有特征维度
    return ucn_adj, features, cross_and_neighbors


def get_node_mask_batch(adj_matrices):
    node_numbers = adj_matrices.shape[1]
    left = np.arange(0, node_numbers, 2)
    right = np.arange(1, node_numbers, 2)

    masks = np.zeros_like(adj_matrices)
    cross_nodes_list = []

    for i in range(len(adj_matrices)):
        adj = adj_matrices[i]

        # --- 步骤1：识别交叉节点 ---
        # 左→右连接（左脑节点是否有连接右脑的边）
        lr_conn = (adj[left][:, right] != 0).any(axis=1)
        # 右→左连接（右脑节点是否有连接左脑的边）
        rl_conn = (adj[right][:, left] != 0).any(axis=1)

        # 合并交叉节点
        cross_nodes = np.concatenate([
            left[lr_conn],
            right[rl_conn]
        ])
        cross_nodes_list.append(cross_nodes)

        # --- 步骤2：生成掩码矩阵 ---
        masks[i, cross_nodes] = 1
        masks[i, :, cross_nodes] = 1

    return masks, cross_nodes_list

def create_data_list(features, adj_matrices, graph_type='x1'):
    data_list = []
    for feat, adj in zip(features, adj_matrices):
        edge_index, edge_weight = dense_to_sparse(adj)
        data = Data(
            x=feat,
            edge_index=edge_index,
            edge_weight=edge_weight,
            graph_type=graph_type
        )
        data_list.append(data)
    return data_list


def divide_brain_network_L_and_R(adj, features):
    L_adj = adj[:, ::2, ::2].copy()
    L_features = features[:, ::2, :].copy()

    R_adj = adj[:, 1::2, 1::2].copy()
    R_features = features[:, 1::2, :].copy()

    return R_adj, R_features, L_adj, L_features


class Metrics:
    def __init__(self):
        self.test_accuracy_list = []
        self.train_accuracy_list = []
        self.all_metrics_list = []

    def calculate_accuracy(self, true_labels, predicted_labels):
        # 计算测试集结果精度，并将该精度添加到test_accuracy_list
        accuracy = accuracy_score(true_labels, predicted_labels)
        logger.warning('Current test accuracy is {}'.format(self.percentage(accuracy)))
        return self.percentage(accuracy)

    def test_calculate_accuracy(self, true_labels, predicted_labels):
        # 计算测试集结果精度，并将该精度添加到test_accuracy_list
        accuracy = accuracy_score(true_labels, predicted_labels)
        self.test_accuracy_list.append(accuracy)
        return self.percentage(accuracy)

    def train_calculate_accuracy(self, true_labels, predicted_labels):
        # 计算训练集结果精度，并将结果添加到train_accuracy_list
        accuracy = accuracy_score(true_labels, predicted_labels)
        self.train_accuracy_list.append(accuracy)
        return self.percentage(accuracy)

    def test_average_accuracy(self):
        if len(self.test_accuracy_list) == 0:
            raise ValueError('test_accuracy_list length is zero.')
        return self.calculate_mean_list(self.test_accuracy_list)

    def train_average_accuracy(self):
        if len(self.train_accuracy_list) == 0:
            raise ValueError('train_accuracy_list length is zero.')
        return self.calculate_mean_list(self.train_accuracy_list)
    def calculate_all_metrics(self, true_labels, predicted_labels, average='weighted'):
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average=average)
        sensitivity = recall_score(true_labels, predicted_labels, average=average)
        precision = precision_score(true_labels, predicted_labels, average=average)
        auc = roc_auc_score(true_labels, predicted_labels, average=average)
        specificity= self.specificity_score(true_labels, predicted_labels)
        self.all_metrics_list.append((accuracy, f1, sensitivity, precision,specificity, auc))
        logger.info('Test accuracy_score: {}%'.format(self.percentage(accuracy)))
        logger.info('Test f1_score: {}%'.format(self.percentage(f1)))
        logger.info('Test recall_score: {}%'.format(self.percentage(sensitivity)))
        logger.info('Test precision_score: {}%'.format(self.percentage(precision)))
        logger.info('Test specificity_score: {}%'.format(self.percentage(specificity)))
        logger.info('Test roc_auc_score: {}%'.format(self.percentage(auc)))
        return accuracy, f1, sensitivity, precision, auc

    def average_all_metrics(self):
        means = [sum(t) / len(t) for t in zip(*self.all_metrics_list)]
        std_devs = [((sum((x - mean) ** 2 for x in t) / len(t)) ** 0.5) for mean, t in zip(means, zip(*self.all_metrics_list))]

        average_list = list(map(lambda x: round(x*100, 2), means))
        average_std = list(map(lambda x: round(x*100, 2), std_devs))

        logger.info(
            f'All average metrics: average-accuracy:{average_list[0]}±{average_std[0]},'
            f' average-f1-score:{average_list[1]}±{average_std[1]},'
            f' average-sensitivity:{average_list[2]}±{average_std[2]},'
            f' average-precision_score:{average_list[3]}±{average_std[3]},'
            f' average-specificity_score:{average_list[4]}±{average_std[4]},'
            f' average-roc_auc_score:{average_list[5]}±{average_std[5]}.')
        return average_list

    def specificity_score(self,true_labels, predicted_labels):
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        specificity = tn / (tn + fp)
        return specificity

    def accuracy_in_all_metrics(self):
        average_list = []
        for idx, (accuracy, f1, sensitivity, precision,specificity, auc) in enumerate(self.all_metrics_list):
            average_list.append(accuracy)
        return average_list

    def accuracy_average_in_all_metrics(self):
        return self.calculate_mean_list(self.accuracy_in_all_metrics())

    def percentage(self,value, decimals=2):
        # 计算value的百分数数，并默认保留两位小数
        return round(100 * value, decimals)

    def calculate_mean_list(self,lst, decimals=2, Percentage: bool = True):
        mean = np.mean(lst)
        if Percentage:
            mean *= 100
        return round(mean, decimals)