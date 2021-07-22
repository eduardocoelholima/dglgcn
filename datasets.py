import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
import dgl
# from dgl._deprecate.graph import DGLGraph
# from .. import backend as F
from normalization import fetch_normalization, row_normalize
from time import perf_counter

class DblpLikeDataset(object):
    def __init__(self, name='imdb', raw_dir=None, force_reload=False, verbose=True, reverse_edge=True):
        # assert name.lower() in ['cora', 'citeseer', 'pubmed']
        # if name.lower() == 'cora':
        #     name = 'cora_v2'
        # url = _get_dgl_url(self._urls[name])
        # self._reverse_edge = reverse_edge
        # name = 'dblp'
        # self.process()
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_dblp_like_data(name)
        np_g = self.adj.cpu().to_dense().numpy()
        sp_g = sp.csr_matrix(np_g)
        self._g = dgl.from_scipy(sp_g)

        self.num_labels = np.unique(self.labels.cpu().numpy())

        # graph = self._g
        # # for compatability
        # graph = graph.clone()
        # graph.ndata.pop('train_mask')
        # graph.ndata.pop('val_mask')
        # graph.ndata.pop('test_mask')
        # graph.ndata.pop('feat')
        # graph.ndata.pop('label')
        # graph = to_networkx(graph)
        # self._graph = nx.DiGraph(graph)
        #
        # self._num_classes = info['num_classes']
        # self._g.ndata['train_mask'] = generate_mask_tensor(F.asnumpy(self._g.ndata['train_mask']))
        # self._g.ndata['val_mask'] = generate_mask_tensor(F.asnumpy(self._g.ndata['val_mask']))
        # self._g.ndata['test_mask'] = generate_mask_tensor(F.asnumpy(self._g.ndata['test_mask']))
        # # hack for mxnet compatability


    # def process(self):
        """Loads input data from data directory and reorder graph for better locality
        """

        # _g = dgl.DGLGraph()

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

def generate_mask_tensor(mask):
    """Generate mask tensor according to different backend
    For torch and tensorflow, it will create a bool tensor
    For mxnet, it will create a float tensor
    Parameters
    ----------
    mask: numpy ndarray
        input mask tensor
    """
    assert isinstance(mask, np.ndarray), "input for generate_mask_tensor" \
        "should be an numpy ndarray"
    if F.backend_name == 'mxnet':
        return F.tensor(mask, dtype=F.data_type_dict['float32'])
    else:
        return F.tensor(mask, dtype=F.data_type_dict['bool'])

def load_data(loadpath = './data/dblp/'):
    with open(loadpath+'labels.pkl', 'rb') as f:
        labels = pkl.load(f)
    with open(loadpath + 'node_features.pkl', 'rb') as f:
        features = pkl.load(f)
    with open(loadpath + 'edges.pkl', 'rb') as f:
        edges = pkl.load(f)
    return labels, features, edges



def load_dblp_like_data(dataset="dblp", normalization="AugNormAdj", cuda=True):
    """
    Load DBLP Networks Datasets.
    """

    # dataset = 'dblp'
    labels_onehot, features, edges = load_data(loadpath='./data/' + dataset + '/')

    metaedges = [edges[1]@edges[0], edges[1]@edges[2]@edges[3]@edges[0]] if dataset == 'dblp' \
        else [edges[0]@edges[1], edges[2]@edges[3]]
    adj = metaedges[0]

    # labels = [np.where(r == 1)[0][0] for r in labels_onehot]
    # for i,l in enumerate(labels[0]):
    train = np.array(labels_onehot[0])
    val = np.array(labels_onehot[1])
    test = np.array(labels_onehot[2])
    idx_train = train[:, 0]
    idx_val = val[:, 0]
    idx_test = test[:, 0]
    labels = np.zeros(features.shape[0])
    labels[idx_train] = train[:,1]
    labels[idx_val] = val[:, 1]
    labels[idx_test] = test[:, 1]
    # labels = np.vstack((train[:,1].reshape(-1,1), val[:,1].reshape(-1,1), test[:,1].reshape(-1,1)))


    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))

    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # test_idx_range = np.sort(test_idx_reorder)

    # if dataset_str == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #
    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    #
    features = torch.FloatTensor(features).float()
    adj, features = preprocess_citation(adj, features, normalization)
    features = torch.FloatTensor(features).float()

    # porting to pytorch
    # features = torch.FloatTensor(np.array(features.todense())).float()
    # features = torch.FloatTensor(features).float()

    # labels = torch.LongTensor(labels)
    labels = torch.LongTensor(labels.reshape(-1))
    # labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # should adj be binary???? metaedges currently are not

    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index
