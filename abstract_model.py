import sys, os
import os.path as osp
import json
import math
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = True

from src import utils
from src.utils import softmax


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, mean=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(2*in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mean_aggr = mean

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ori_input, input, adj):
        h = torch.spmm(adj, input)
        h1 = torch.cat((h, ori_input), dim=1)
        output = torch.mm(h1, self.weight)
        if self.mean_aggr:
            output /= adj.sum(dim=1, keepdim=True)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModel(nn.Module):
    def __init__(self, node_emd_init, taxo_emd_init, dropout=0.0):
        super(GCNModel, self).__init__()
        self.emb_size, self.emb_dimension = node_emd_init.shape
        self.node_embed = nn.Embedding.from_pretrained(torch.from_numpy(node_emd_init), freeze=False)
        self.taxo_embed = nn.Embedding.from_pretrained(torch.from_numpy(taxo_emd_init), freeze=False)
        self.dropout = dropout
        self.encoder = GraphConvolution(self.emb_dimension, self.emb_dimension, mean=True)
        self.double()

    def forward(self, nodes, ori_nodes, categorys, adj, label):
        node_embed = self.node_embed(nodes)
        ori_node_embed = self.node_embed(ori_nodes)
        h = F.relu(self.encoder(ori_node_embed, node_embed, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        # output = self.decoder(h.reshape(-1, 2*self.emb_dimension)).squeeze()
        # l, r = h.reshape(-1, 2*self.emb_dimension).split(self.emb_dimension, dim=1)
        # output = (l*r).sum(dim=1)
        taxo_embed = self.taxo_embed(categorys)
        score = (h * taxo_embed).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(score, label)
        return loss

    def get_embed(self):
        return self.node_embed.weight.data.cpu().numpy()

    def get_taxo_embed(self):
        return self.taxo_embed.weight.data.cpu().numpy()


class TaxoGCNModel(GCNModel):
    def __init__(self, node_emd_init, taxo_emd_init, dropout=0.0):
        super(TaxoGCNModel, self).__init__(node_emd_init, taxo_emd_init, dropout)

    def forward_node(self, nodes, ori_nodes, adj, label):
        node_embed = self.node_embed(nodes)
        ori_node_embed = self.node_embed(ori_nodes)
        h = F.relu(self.encoder(ori_node_embed, node_embed, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        l, r = h.reshape(-1, 2 * self.emb_dimension).split(self.emb_dimension, dim=1)
        output = (l * r).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(output, label)
        return loss


class SkipGramModel(nn.Module):
    def __init__(self, node_emd_init):
        super(SkipGramModel, self).__init__()
        self.emb_size, self.emb_dimension = node_emd_init.shape
        self.w_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(node_emd_init), freeze=False)
        self.v_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(node_emd_init), freeze=False)
        self.double()

    def forward(self, nodes, labels):
        emb_w = self.w_embeddings(nodes[:, 0])  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        emb_v = self.v_embeddings(nodes[:, 1])
        score = (emb_w*emb_v).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss

    def get_embed(self):
        return self.w_embeddings.weight.data.cpu().numpy()


class TaxoSkipGramModel(SkipGramModel):
    def __init__(self, node_emd_init, taxo_emd_init):
        super(TaxoSkipGramModel, self).__init__(node_emd_init)
        self.emb_size, self.emb_dimension = node_emd_init.shape
        self.taxo_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(taxo_emd_init), freeze=False)
        self.double()

    def forward_taxo(self, nodes, labels):
        emb_w = self.w_embeddings(nodes[:, 0])
        emb_t = self.taxo_embeddings(nodes[:, 1])
        score = (emb_w*emb_t).sum(dim=1)
        loss = (F.binary_cross_entropy_with_logits(score, labels)).mean()
        return loss

    def get_taxo_embed(self):
        return self.taxo_embeddings.weight.data.cpu().numpy()

    def return_node_embedding_by_path(self, node, level):
        return self.w_embeddings.weight[node].data.cpu().numpy().reshape(1, -1).repeat(level, axis=0)


class AbstractClass(object):
    def __init__(self, args, logger=None):
        """initialize the parameters, prepare the data and build the network"""
        self.print = logger.info if logger else print
        self.device = args.device

        self.print(f"Reading data from {args.data_dir}")
        self.root_nodes, self.graph, num_link = utils.read_graph(args.link_file)
        self.pre_train_embedding, self.id2name, self.name2id = utils.read_nodes(args.node_file)
        self.num_node = len(self.id2name)
        self.embed_dim = args.embed_dim

        self.print(f"num of link: {num_link}")
        self.print(f"num of node: {self.num_node}")

        if args.rand_init or not self.pre_train_embedding:
            self.print("randomly initialize node embedding!")
            self.embed_dim = args.embed_dim
            self.pre_train_embedding = utils.init_embed(self.num_node, self.embed_dim)

        if 'taxonomy' in args.task:
            self.taxo_parent2children, self.taxo_child2parents, _, self.category2nodeid, self.category2id, _ \
                = utils.read_taxos(args.taxo_file, args.taxo_assign_file, args.extend_label)
            self.taxo_rootid = self.category2id['root']

        self.task_stats = {task: {'best_acc': -1, 'acc': []} for task in args.task.split('|')}
        self.stats = {'graph_loss': [], 'taxo_loss':[]}
        self.model = None

    def train(self, args, evaluate_funcs):
        pass

    def evaluate(self, args, evaluate_funcs, epoch=-1, optimizer=None):
        embed = self.model.get_embed()
        flag = 0
        for task, evaluate_func in evaluate_funcs.items():
            if task == 'taxonomy':
                acc, std = evaluate_func(self)
            else:
                acc, std = evaluate_func(embed)

            if epoch >= 0 and optimizer:
                self.task_stats[task]['acc'].append((epoch, acc, std))

                if acc > self.task_stats[task]['best_acc']:
                    self.task_stats[task]['best_acc'] = acc
                    self.task_stats[task]['best_std'] = std
                    self.task_stats[task]['best_epoch'] = epoch
                    self.task_stats[task]['best_model'] = self.model.state_dict()
                    self.task_stats[task]['best_opt'] = optimizer.state_dict()
                    filename, file_extension = os.path.splitext(args.embed_path)
                    embed_path = filename + f'.{task}' + file_extension
                    try:
                        self.save_embedding(embed_path, embed)
                    except OSError as e:
                        self.print(e)
                    self.task_stats[task]['count'] = 0
                else:
                    if args.early_stop and epoch>args.minimal_epoch:
                        self.task_stats[task]['count'] += args.save_every
                        if self.task_stats[task]['count'] >= args.patience:
                            self.print('early stopped!')
                            flag += 1
                best_acc = self.task_stats[task]['best_acc']
                best_std = self.task_stats[task]['best_std']
                best_epoch = self.task_stats[task]['best_epoch']
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}] acc={acc:.2f} += {std:.2f}, BEST: acc={best_acc:.2f} += {best_std:.2f} @ {best_epoch}')
            else:
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}] acc={acc:.2f} += {std:.2f}')
        return flag == len(evaluate_funcs)

    def find_category(self, node_embeddings, categorys=None):
        taxo_embed = self.init_taxo_embed(self.model.get_embed())
        if categorys is None:
            score = node_embeddings @ taxo_embed.T
            assignments = score.argmax(axis=1)
            for i, chosen in enumerate(assignments):
                children = self.taxo_parent2children.get(chosen, [])
                while children:
                    chosen = np.argmax(node_embeddings[i] @ taxo_embed[children].T)
                    children = self.taxo_parent2children.get(chosen, [])
                assignments[i] = chosen
        else:
            cateids = [self.category2id[c] for c in categorys]
            score = node_embeddings @ taxo_embed[cateids].T
            assignments = [categorys[i] for i in score.argmax(axis=1)]
        return assignments

    def evaluate_taxo(self, test_data, level_by_level=True, split=5):
        correct = []
        node_embed = self.model.get_embed()
        taxo_embed = self.init_taxo_embed(node_embed)
        for (node, category) in test_data:
            category_path = self.find_path(self.category2id[category])[:-1]
            level = len(category_path)
            embeds = node_embed[node].reshape(1, -1).repeat(level, axis=0)
            p = self.taxo_rootid
            for embed, true_pos in zip(embeds, category_path[::-1]):
                candidates = self.taxo_parent2children.get(p, None)
                if candidates is None:
                    correct.append(False)
                    continue
                pred = (embed @ taxo_embed[candidates].T).argmax()
                pred = candidates[pred]
                correct.append(pred == true_pos)
                if level_by_level:
                    p = true_pos
                else:
                    p = pred
        accs = np.array([np.sum(i)*100/len(i) for i in np.array_split(correct, split)])
        return accs.mean(), accs.std()

    def find_path(self, c):
        path = [c]
        while True:
            p = self.taxo_child2parents.get(c, None)
            if p is None:
                return path
            else:
                path.append(p)
                c = p

    def make_dist(self, power=0.75):
        edgedistdict = collections.defaultdict(int)
        taxodistdict = collections.defaultdict(int)
        weightsum = 0
        negprobsum = 0
        # can work with weighted edge, but currently we do not have
        weight = 1
        for node, cs in self.nodeid2category.items():
            for c in cs:
                edgedistdict[tuple([node, c])] = weight
                weightsum += weight
                negprobsum += np.power(weight, power)
                taxodistdict[c] += weight

        for node, outdegree in taxodistdict.items():
            taxodistdict[node] = np.power(outdegree, power) / negprobsum

        for edge, weight in edgedistdict.items():
            edgedistdict[edge] = weight / weightsum

        return edgedistdict, taxodistdict

    def init_taxo_embed(self, node_init_embedding):
        default = np.mean(node_init_embedding, axis=0)
        init_taxo_embed = np.repeat(default.reshape(1, -1), repeats=(len(self.category2id)), axis=0)
        root = self.category2id['root']
        visited, queue = [root], collections.deque([root])
        while queue:
            vertex = queue.popleft()
            neighbour = self.taxo_parent2children.get(vertex, [])
            visited += neighbour
            queue += neighbour
        # assert len(set(visited))==len(visited)
        # make sure child node being initialized first
        init_list = visited[::-1]
        for i, cid in enumerate(init_list):
            children = self.taxo_parent2children.get(cid, [])
            if children:
                children = self.taxo_parent2children[cid]
                init_taxo_embed[cid] = init_taxo_embed[children].mean()
            else:
                nodes = self.category2nodeid.get(cid, [])
                if nodes:
                    init_taxo_embed[cid] = node_init_embedding[nodes].mean()
        return init_taxo_embed

    def save_all(self, args):
        self.print('='*50+'Done'+'='*50)
        flag = 1
        try:
            json.dump(self.stats, open(osp.join(args.log_dir, 'stats.json'), 'w'), indent=4)
            for task, stat in self.task_stats.items():
                best_epoch = stat['best_epoch']
                best_acc = stat['best_acc']
                best_std = stat['best_std']
                self.save_checkpoint({
                    'args': args,
                    'model': stat.pop('best_model'),
                    'optimizer': stat.pop('best_opt')
                }, args.log_dir, f'epoch{best_epoch}_acc{best_acc:.2f}_task{task}.pth.tar', flag)
                self.print(f'[BEST-EVALUATION:{args.model} {args.dataset} {task}] best acc ={best_acc:.2f} +- {best_std:.2f} @epoch:{best_epoch:d}')
                flag = 0
            json.dump(self.task_stats, open(osp.join(args.log_dir, 'task_stats.json'), 'w'), indent=4)
        except OSError as e:
            self.print(e)
            for task, stat in self.task_stats.items():
                best_epoch = stat['best_epoch']
                best_acc = stat['best_acc']
                best_std = stat['best_std']
                self.print(f'[BEST-EVALUATION:{args.model} {args.dataset} {task}] best acc ={best_acc:.2f} +- {best_std:.2f} @epoch:{best_epoch:d}')

    def save_embedding(self, path, embed):
        node_embed_list = embed.tolist()
        node_embed_str = [name+"\t"+"\t".join([str(x) for x in line]) for name, line in zip(self.id2name, node_embed_list)]
        with open(path, "w") as f:
            lines = [str(len(node_embed_str)) + "\t" + str(self.embed_dim)] + node_embed_str
            f.write('\n'.join(lines))

    def save_checkpoint(self, state, modelpath, modelname, del_others=True):
        if del_others:
            for dirpath, dirnames, filenames in os.walk(modelpath):
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    if path.endswith('pth.tar'):
                        self.print(f'rm {path}')
                        os.system("rm -rf '{}'".format(path))
                break
        path = os.path.join(modelpath, modelname)
        self.print('saving model to {}...'.format(path))
        torch.save(state, path)


class AbstractGAN(AbstractClass):
    def __init__(self, args, logger=None):
        """initialize the parameters, prepare the data and build the network"""
        super(AbstractGAN, self).__init__(args, logger)


        # build the model
        self.generator = None
        self.discriminator = None
        self.all_score = None
        self.task_stats = {task: {'best_acc': -1, 'best_acc_d': -1, 'best_acc_g': -1, 'd_acc': [], 'g_acc': []} for task in args.task.split('|')}
        self.stats = {'d_g_loss': [], 'd_t_loss': [], 'g_g_loss': [], 'g_t_loss': []}

        edge_dist_dict, node_dist_dict = utils.makeDist(args.link_file, args.negative_power)
        # self.edges_alias_sampler = utils.VoseAlias(edge_dist_dict)
        self.nodes_alias_sampler = utils.VoseAlias(node_dist_dict)

    def train(self, args, evaluate_func):
        pass

    def evaluate(self, args, evaluate_funcs, epoch=-1, d_optimizer=None, g_optimizer=None):
        d_embed = self.discriminator.get_embed()
        g_embed = self.generator.get_embed()
        flag = 0
        for task, evaluate_func in evaluate_funcs.items():
            if task == 'taxonomy':
                self.eval_model = self.discriminator
                d_acc, d_std = evaluate_func(self)
                self.eval_model = self.generator
                g_acc, g_std = evaluate_func(self)
            else:
                d_acc, d_std = evaluate_func(d_embed)
                g_acc, g_std = evaluate_func(g_embed)

            if epoch >= 0 and d_optimizer and g_optimizer:
                self.task_stats[task]['d_acc'].append((epoch, d_acc, d_std))
                self.task_stats[task]['g_acc'].append((epoch, g_acc, g_std))
                if d_acc > self.task_stats[task]['best_acc_d']:
                    self.task_stats[task]['best_acc_d'] = d_acc
                    self.task_stats[task]['best_std_d'] = d_std
                    self.task_stats[task]['best_epoch_d'] = epoch
                    self.task_stats[task]['best_model_d'] = self.discriminator.state_dict()
                    self.task_stats[task]['best_opt_d'] = d_optimizer.state_dict()
                    filename, file_extension = os.path.splitext(args.embed_path)
                    embed_path = filename + f'.d.{task}' + file_extension
                    try:
                        self.save_embedding(embed_path, d_embed)
                    except OSError as e:
                        self.print(e)

                if g_acc > self.task_stats[task]['best_acc_g']:
                    self.task_stats[task]['best_acc_g'] = g_acc
                    self.task_stats[task]['best_std_g'] = g_std
                    self.task_stats[task]['best_epoch_g'] = epoch
                    self.task_stats[task]['best_model_g'] = self.generator.state_dict()
                    self.task_stats[task]['best_opt_g'] = g_optimizer.state_dict()
                    filename, file_extension = os.path.splitext(args.embed_path)
                    embed_path = filename + f'.g.{task}' + file_extension
                    try:
                        self.save_embedding(embed_path, g_embed)
                    except OSError as e:
                        self.print(e)

                if d_acc > g_acc:
                    test_acc = d_acc
                    std = d_std
                else:
                    test_acc = g_acc
                    std = g_std
                if test_acc > self.task_stats[task]['best_acc']:
                    self.task_stats[task]['best_acc'] = test_acc
                    self.task_stats[task]['best_std'] = std
                    self.task_stats[task]['best_epoch'] = epoch
                    self.task_stats[task]['count'] = 0
                    # self.save_all(args)
                else:
                    if args.early_stop and epoch>args.minimal_epoch:
                        self.task_stats[task]['count'] += args.save_every
                        if self.task_stats[task]['count'] >= args.patience:
                            self.print('early stopped!')
                            flag += 1
                best_acc = self.task_stats[task]['best_acc']
                best_std = self.task_stats[task]['best_std']
                best_epoch = self.task_stats[task]['best_epoch']
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}] d acc={d_acc:.2f} += {d_std:.2f} g acc={g_acc:.2f} += {g_std:.2f} BEST: acc={best_acc:.2f} += {best_std:.2f} @ {best_epoch}')
            else:
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}] d acc={d_acc:.2f} += {d_std:.2f} g acc={g_acc:.2f} += {g_std:.2f}')
        return flag == len(evaluate_funcs)

    def find_category(self, node_embeddings, categorys=None):
        taxo_embed = self.eval_model.get_taxo_embed()
        if categorys is None:
            score = node_embeddings @ taxo_embed.T
            assignments = score.argmax(axis=1)
            for i, chosen in enumerate(assignments):
                children = self.taxo_parent2children.get(chosen, [])
                while children:
                    chosen = np.argmax(node_embeddings[i] @ taxo_embed[children].T)
                    children = self.taxo_parent2children.get(chosen, [])
                assignments[i] = chosen
        else:
            cateids = [self.category2id[c] for c in categorys]
            score = node_embeddings @ taxo_embed[cateids].T
            assignments = [categorys[i] for i in score.argmax(axis=1)]
        return assignments

    def evaluate_taxo(self, test_data, level_by_level=True, split=5):
        correct = []
        taxo_embed = self.eval_model.get_taxo_embed()
        taxo_bias = self.eval_model.get_taxo_bias()
        transforms = self.eval_model.get_transforms()
        for (node, category) in test_data:
            category_path = self.find_path(self.category2id[category])[:-1]
            level = len(category_path)
            embeds = self.eval_model.return_node_embedding_by_path(node, transforms, level)
            p = self.taxo_rootid
            for embed, true_pos in zip(embeds, category_path[::-1]):
                candidates = self.taxo_parent2children.get(p, None)
                if candidates is None:
                    correct.append(False)
                    continue
                pred = (embed @ taxo_embed[candidates].T + taxo_bias[candidates]).argmax()
                pred = candidates[pred]
                correct.append(pred == true_pos)
                if level_by_level:
                    p = true_pos
                else:
                    p = pred
        accs = np.array([np.sum(i)*100/len(i) for i in np.array_split(correct, split)])
        return accs.mean(), accs.std()

    def update_all_score(self):
        self.all_score = self.generator.get_all_score()

    def get_batch_data(self, left, right, label, batch_size, convert=True):
        if convert:
            for start in range(0, len(label), batch_size):
                end = start + batch_size
                yield torch.LongTensor(left[start:end]).to(self.device), torch.LongTensor(right[start:end]).to(self.device), torch.DoubleTensor(label[start:end]).to(self.device)
        else:
            for start in range(0, len(label), batch_size):
                end = start + batch_size
                yield left[start:end], right[start:end], label[start:end]

    def sample(self, root, sample_num, sample_for_dis: bool):
        all_score = self.all_score

        if sample_for_dis:
            root_neighbor = self.graph[root]
            k = min(int(0.1 * len(self.root_nodes)), sample_num*len(root_neighbor))
            sampled_nodes = [i for i in random.sample(self.root_nodes, k=k) if
                             i not in root_neighbor]
            if len(sampled_nodes) == 0:
                return []

            prob = softmax(all_score[root, sampled_nodes])
            sample = np.random.choice(sampled_nodes, size=sample_num, p=prob).tolist()
            return sample
        else:
            trace = [root]
            node_select = root
            while True:
                node_neighbor = self.graph[node_select]
                prob = softmax(all_score[node_select, node_neighbor])
                node_select = np.random.choice(node_neighbor, size=1, p=prob)[0]
                trace.append(node_select)
                if len(trace) == sample_num:
                    return trace

    def sample_for_d(self, data_size, n_sample_d, update_ratio=1.0):
        center_nodes = []
        neighbor_nodes = []
        labels = []
        while len(labels) < data_size:
            u = self.nodes_alias_sampler.alias_generation()
        # for u in self.root_nodes:
        #     if np.random.rand() < update_ratio:
            pos = self.graph[u]
            n_pos = len(pos)
            if n_pos < 1:
                continue
            else:
                if n_pos > n_sample_d:
                    pos = random.sample(pos, n_sample_d)
            neg = self.sample(u, len(pos), sample_for_dis=True)
            # if len(neg) <= len(pos):
            #     pos = random.sample(pos, len(neg))
            # else:
            #     neg = random.sample(neg, len(pos))
            neighbors = pos + neg
            center_nodes.extend(len(neighbors) * [u])
            neighbor_nodes.extend(neighbors)
            labels.extend(len(pos) * [1] + len(neg) * [0])
        return center_nodes, neighbor_nodes, labels

    def sample_for_g(self, data_size, n_sample_g, window_size, update_ratio=1.0):
        left, right = [], []
        while len(left) < data_size:
            root_node = self.nodes_alias_sampler.alias_generation()
            trace = self.sample(root_node, n_sample_g, sample_for_dis=False)

            path = trace[:-1]
            for i, center_node in enumerate(path):
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(path))):
                    if i != j:
                        left.append(center_node)
                        right.append(path[j])
        return left, right

    def generate_window_pairs(self, paths, window_size):
        left, right = [], []
        for path in paths:
            path = path[:-1]
            for i, center_node in enumerate(path):
                for j in range(max(i - window_size, 0), min(i + window_size + 1, len(path))):
                    if i != j:
                        left.append(center_node)
                        right.append(path[j])
        return left, right

    def save_all(self, args):
        self.print('=' * 50 + 'Done' + '=' * 50)
        log_dir = args.log_dir
        flag = 1
        try:
            json.dump(self.stats, open(osp.join(log_dir, 'stats.json'), 'w'), indent=4)
            for task, stat in self.task_stats.items():
                best_epoch = stat['best_epoch']
                best_acc = stat['best_acc']
                best_epoch_g = stat['best_epoch_g']
                best_acc_g = stat['best_acc_g']
                best_std_g = stat['best_std_g']
                best_epoch_d = stat['best_epoch_d']
                best_acc_d = stat['best_acc_d']
                best_std_d = stat['best_std_d']
                self.save_checkpoint({
                    'args': args,
                    'd_model': stat.pop('best_model_d'),
                    'd_optimizer': stat.pop('best_opt_d'),
                    'g_model': stat.pop('best_model_g'),
                    'g_optimizer': stat.pop('best_opt_g')
                }, log_dir, f'epoch{best_epoch}_acc{best_acc:.2f}_task{task}.pth.tar', flag)
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}]: best acc g ={best_acc_g:.2f} +- {best_std_g:.2f} @epoch:{best_epoch_g:d}')
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}]: best acc d ={best_acc_d:.2f} +- {best_std_d:.2f} @epoch:{best_epoch_d:d}')
                flag = 0
            json.dump(self.task_stats, open(osp.join(log_dir, 'task_stats.json'), 'w'), indent=4)
        except OSError as e:
            self.print(e)
            for task, stat in self.task_stats.items():
                best_epoch_g = stat['best_epoch_g']
                best_acc_g = stat['best_acc_g']
                best_std_g = stat['best_std_g']
                best_epoch_d = stat['best_epoch_d']
                best_acc_d = stat['best_acc_d']
                best_std_d = stat['best_std_d']
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}]: best acc g ={best_acc_g:.2f} +- {best_std_g:.2f} @epoch:{best_epoch_g:d}')
                self.print(f'[EVALUATION:{args.model} {args.dataset} {task}]: best acc d ={best_acc_d:.2f} +- {best_std_d:.2f} @epoch:{best_epoch_d:d}')