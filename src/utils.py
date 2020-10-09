import os
import csv
import json
import random
import itertools
import collections
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from decimal import *
import torch
from src.logger import myLogger
# matplotlib.use('Agg')


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())


def mean_skip_zero(tensor):
    return tensor.sum()/(tensor!=0).sum()


def init_embed(n, dim):
    return np.random.uniform(-0.5 / dim, 0.5 / dim, size=(n, dim))


def l2_loss(tensor):
    return 0.5*((tensor ** 2).sum())


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numberation stablity
    return e_x / e_x.sum()


def print_config(config, logger=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        to_add = "\t{} : {}\n".format(k, str(v))
        if len(to_add) < 1000:
            info += to_add
    info.rstrip()
    if not logger:
        print("\n" + info)
    else:
        logger.info("\n" + info)


def init_logger(args):
    if args.prefix:
        base = os.path.join('log', args.prefix)
        log_dir = os.path.join(base, args.suffix)
    else:
        tag = f'{args.model}_{args.dataset}_{args.task}_'
        comment = args.suffix if args.suffix else datetime.now().strftime('%b_%d_%H-%M-%S')
        log_dir = os.path.join('log', tag + comment)
    args.log_dir = log_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = myLogger(name='exp', log_path=os.path.join(log_dir, 'log.txt'))
    logger.setLevel(args.log_level)
    return logger


def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func


def save_checkpoint(state, modelpath, modelname, logger=None, del_others=True):
    if del_others:
        for dirpath, dirnames, filenames in os.walk(modelpath):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if path.endswith('pth.tar'):
                    if logger is None:
                        print(f'rm {path}')
                    else:
                        logger.warning(f'rm {path}')
                    os.system("rm -rf '{}'".format(path))
            break
    path = os.path.join(modelpath, modelname)
    if logger is None:
        print('saving model to {}...'.format(path))
    else:
        logger.warning('saving model to {}...'.format(path))
    torch.save(state, path)


def read_graph(link_file):
    """read data from files

    Args:
        link_file: link file path

    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = collections.defaultdict(set)
    nodes = set()
    num_link = 0
    with open(link_file) as fin:
        for l in fin:
            num_link += 1
            n1, n2 = map(int, l.strip().split('\t'))
            nodes.add(n1)
            nodes.add(n2)
            graph[n1].add(n2)
            graph[n2].add(n1)
    graph = {k:list(v) for k, v in graph.items()}
    return list(nodes), graph, num_link


def read_nodes(node_file):
    """read pretrained node embeddings
    """

    id2name = []
    name2id = {}
    embedding_matrix = None
    with open(node_file, "r") as fin:
        vecs = []
        for l in fin:
            l = l.strip().split('\t')
            if len(l)==2:
                name, embed = l
                vecs.append([float(i) for i in embed.split(',')])
            else:
                name = l[0]
            id2name.append(name)
            name2id[name] = id
        if len(vecs)==len(id2name):
            embedding_matrix = np.array(vecs)
    return embedding_matrix, id2name, name2id


def sublist(lst1, lst2):
    # whether lst1 is sublist of lst2
    return set(lst1) <= set(lst2)


def read_taxos(taxo_file, taxo_assign_file, extend_label=True):
    """read taxonomy assignment
    """
    taxo_parent2children = json.load(open(taxo_file))['parent2children']
    categories = set()
    for p, cs in taxo_parent2children.items():
        categories.add(p)
        categories.update(cs)
    category2id = {c: i for i, c in enumerate(sorted(list(categories)))}
    taxo_parent2children = {category2id[p]:[category2id[c] for c in cs] for p, cs in taxo_parent2children.items()}
    taxo_child2parents = {c: p for p, cs in taxo_parent2children.items() for c in cs}

    nodeid2category = collections.defaultdict(list)
    with open(taxo_assign_file, "r") as fin:
        for l in fin:
            nodeid, category = l.strip().split('\t')
            nodeid = int(nodeid)
            cateid = category2id[category]
            nodeid2category[nodeid].append(cateid)
    category2id = dict(category2id)

    nodeid2path = {}
    for node, categorys in nodeid2category.items():
        paths = []
        for c in categorys:
            path = collections.deque()
            while True:
                p = taxo_child2parents.get(c, None)
                if p is None:
                    break
                path.appendleft(c)
                c = p
            paths.append(list(path))
        nodeid2path[node] = paths

    # remove sub-path
    new_nodeid2path = {}
    for nodeid, paths in nodeid2path.items():
        if len(paths) == 1:
            unique_paths = paths
        else:
            unique_paths = []
            for i, path in enumerate(paths):
                flag = 1
                for j, other_path in enumerate(paths):
                    if i==j:
                        continue
                    if sublist(path, other_path):
                        flag = 0
                        break
                if flag:
                    unique_paths.append(path)
        new_nodeid2path[nodeid] = unique_paths
    nodeid2path = new_nodeid2path

    category2nodeid = collections.defaultdict(list)
    if extend_label:
        nodeid2category = {}
        for nodeid, paths in nodeid2path.items():
            cs = list(set([i for p in paths for i in p]))
            nodeid2category[nodeid] = cs
            for c in cs:
                category2nodeid[c].append(nodeid)
    else:
        for nodeid, category in nodeid2category.items():
            for c in category:
                category2nodeid[c].append(nodeid)
    category2nodeid = dict(category2nodeid)

    return taxo_parent2children, taxo_child2parents, nodeid2category, category2nodeid, category2id, nodeid2path


def category2level(categories, child2parent):
    category2level = {}
    for category in categories:
        path = []
        c = category
        while True:
            p = child2parent.get(c, None)
            if p is None:
                category2level[category] = len(path)
                break
            else:
                path.append(p)
                c = p
    return category2level


def read_emd(filename, n_node, n_embed):
    """use the pretrain node embeddings
    """
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
    node_embed = np.random.rand(n_node, n_embed)
    for line in lines:
        emd = line.split()
        node_embed[int(float(emd[0])), :] = str_list_to_float(emd[1:])
    return node_embed


def build_adj(graph, num_node):
    row_id_list = []
    col_id_list = []
    data_list = []
    for node, neighbors in graph.items():
        for n in neighbors:
            row_id_list.append(node)
            col_id_list.append(n)
            data_list.append(1)
    dim = num_node
    return sp.csr_matrix((data_list, (row_id_list, col_id_list)), shape=(dim, dim))


def construct_eval_file_for_taxo_assignment(taxo_file, taxo_assign_file, sample_size=100, min_count=300):
    print('Constructing eval file for taxo assignment')

    taxo_parent2children, taxo_child2parents, nodeid2category, category2nodeid, category2id, nodeid2path = read_taxos(taxo_file, taxo_assign_file)

    leaf_nodes = set(taxo_child2parents.keys()) - set(taxo_parent2children.keys())
    test_category2nodes = {}
    train_category2nodes = {}
    for leaf in leaf_nodes:
        nodes = [n for n in category2nodeid.get(leaf, []) if len(nodeid2path[n])==1]
        if len(nodes)>=min_count:
            sampled = np.random.choice(nodes, size=sample_size*2, replace=False)
            test_category2nodes[leaf] = sampled[:sample_size]
            train_category2nodes[leaf] = sampled[sample_size:]
    test_nodes = [i for _,v in test_category2nodes.items() for i in v]
    train_nodes = [i for _,v in train_category2nodes.items() for i in v]
    print(f'num chosen category: {len(test_category2nodes)}, num train nodes: {len(train_nodes)}, num test nodes: {len(test_nodes)}')

    data_dir = os.path.dirname(taxo_assign_file)
    label = os.path.join(data_dir, 'label.ta.dat')
    id2category = {v:k for k, v in category2id.items()}
    with open(label, 'w') as fout:
        fout.write('train\n')
        for cid, nodes in train_category2nodes.items():
            cate = id2category[cid]
            for node in nodes:
                fout.write(f'{node}\t{cate}\n')
        fout.write('test\n')
        for cid, nodes in test_category2nodes.items():
            cate = id2category[cid]
            for node in nodes:
                fout.write(f'{node}\t{cate}\n')

    remain_taxo = os.path.join(data_dir, 'taxo.remain.ta.dat')
    with open(remain_taxo, 'w') as fout, open(taxo_assign_file) as fin:
        for l in fin:
            nodeid, category = l.strip().split('\t')
            if int(nodeid) not in test_nodes:
                fout.write(l)

    return label, remain_taxo


def construct_eval_file_for_taxo_evaluation(taxo_file, taxo_assign_file, sample_ratio=0.1):
    print('Constructing eval file for taxo evaluation')

    taxo_parent2children, taxo_child2parents, nodeid2category, category2nodeid, category2id, nodeid2path = read_taxos(taxo_file, taxo_assign_file, extend_label=False)

    nodes = list(nodeid2category.keys())
    random.shuffle(nodes)
    sampled = []
    sample_size = int(sample_ratio*len(nodeid2category))
    for node in nodes:
        flag = 1
        paths = nodeid2path[node]
        if len(paths) == 1:
            for c in paths[0][:-1]:
                if len(taxo_parent2children[c])<=1:
                    flag = 0
                    break
            if flag:
                sampled.append(node)
                if len(sampled) == sample_size:
                    break

    data_dir = os.path.dirname(taxo_assign_file)
    label = os.path.join(data_dir, 'label.taxo.dat')
    id2category = {v:k for k, v in category2id.items()}
    with open(label, 'w') as fout:
        for node in sampled:
            c = nodeid2category[node][0]
            fout.write(f'{node}\t{id2category[c]}\n')

    remain_taxo = os.path.join(data_dir, 'taxo.remain.taxo.dat')
    with open(remain_taxo, 'w') as fout, open(taxo_assign_file) as fin:
        for l in fin:
            nodeid, category = l.strip().split('\t')
            if int(nodeid) not in sampled:
                fout.write(l)

    return label, remain_taxo


def construct_eval_file_for_link_prediction(link_file, ratio=0.3):
    print('Constructing eval file for link prediction')

    graph = collections.defaultdict(set)
    with open(link_file) as fin:
        links = set()
        nodes = set()
        for l in tqdm(fin):
            n1, n2 = map(int, l.strip().split('\t'))
            graph[n1].add(n2)
            graph[n2].add(n1)
            if n1 < n2:
                pair = (n1, n2)
            else:
                pair = (n2, n1)
            links.add(pair)
            nodes.update(pair)
    nodes = list(nodes)
    graph = {k:list(v) for k, v in graph.items()}
    num_hidden = int(len(links) * ratio)

    node2degree = {node:len(neighbor) for node, neighbor in graph.items()}
    pos = set()
    flag = 1
    while flag:
        for node, neighbor in tqdm(graph.items()):
            if node2degree[node]!=1:
                n1 = random.choice(neighbor)
                if node2degree[n1]!=1:
                    if node < n1:
                        pair = (node, n1)
                    else:
                        pair = (n1, node)
                    cur_num = len(pos)
                    pos.add(pair)
                    if len(pos) > cur_num:
                        node2degree[node]-=1
                        node2degree[n1]-=1
                        if len(pos) == num_hidden:
                            flag = 0
                            break

    remain = links - pos
    links_list = list(links)

    neg = set()
    while len(neg) < num_hidden:
        n1, n2 = random.choice(links_list)
        n3 = random.choice(nodes)
        if n1 < n3:
            pair = (n1, n3)
        else:
            pair = (n3, n1)
        if pair not in links:
            neg.add(pair)
        n4 = random.choice(nodes)
        if n2 < n4:
            pair = (n2, n4)
        else:
            pair = (n4, n2)
        if pair not in links:
            neg.add(pair)

    data_dir = os.path.dirname(link_file)
    label = os.path.join(data_dir, 'label.lp.dat')
    remain_link = os.path.join(data_dir, 'link.remain.lp.dat')
    with open(label, 'w') as fout:
        for (n1, n2) in pos:
            fout.write(f'{n1}\t{n2}\t1\n')
        for (n1, n2) in neg:
            fout.write(f'{n1}\t{n2}\t0\n')
    with open(remain_link, 'w') as fout:
        for (n1, n2) in remain:
            fout.write(f'{n1}\t{n2}\n')

    return label, remain_link


def construct_feature(data, w2v):
    Data = []
    labels = []
    for word, label in data:
        vector = w2v[word]
        Data.append(vector)
        labels.append(label)
    Data = np.concatenate((np.array(Data), np.array(labels)[:, np.newaxis]), axis=1)
    return Data


def join_int(l):
    return ','.join([str(i) for i in l])


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def makeDist(graphpath, power=0.75):
    edgedistdict = collections.defaultdict(int)
    nodedistdict = collections.defaultdict(int)

    weightsum = 0
    negprobsum = 0
    with open(graphpath, "r") as graphfile:
        # can work with weighted edge, but currently we do not have
        weight = 1
        for l in graphfile:
            line = l.rstrip().split('\t')
            node1, node2 = int(line[0]), int(line[1])
            edgedistdict[tuple([node1, node2])] = weight
            nodedistdict[node1] += weight
            weightsum += weight
            negprobsum += np.power(weight, power)

    for node, outdegree in nodedistdict.items():
        nodedistdict[node] = np.power(outdegree, power) / negprobsum

    for edge, weight in edgedistdict.items():
        edgedistdict[edge] = weight / weightsum

    return edgedistdict, nodedistdict


class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, dist):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}   # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}       # scaled probabilities
        small = []             # stack for probabilities smaller that 1
        large = []             # stack for probabilities greater than or equal to 1

        # Construct and sort the scaled probabilities into their appropriate stacks
        # print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in self.dist.items():
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        # print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        for i in range(size):
            yield self.alias_generation()

    def sample_from(self, candidates, size):
        probs = np.array([self.dist[i] for i in candidates]) + 1e-10
        probs /= np.linalg.norm(probs, ord=1)
        return np.random.choice(candidates, size, p=probs).tolist()

