import os
import os.path as osp
import json, pickle
import time
import random
import functools
import numpy as np
from collections import deque, defaultdict
from tqdm import tqdm, trange
import torch
import torch.optim as optim

from src import utils
from src.abstract_model import SkipGramModel, AbstractClass


class HuffmanNode:
    def __init__(self, word_id, frequency):
        self.word_id = word_id  # 叶子结点存词对应的id, 中间节点存中间节点id
        self.frequency = frequency  # 存单词频次
        self.left_child = None
        self.right_child = None
        self.father = None
        self.Huffman_code = []  # 霍夫曼码（左1右0）
        self.path = []  # 根到叶子节点的中间节点id


class HuffmanTree:
    def __init__(self, wordid_frequency_dict):
        self.word_count = len(wordid_frequency_dict)  # 单词数量
        self.wordid_code = dict()
        self.wordid_path = dict()
        self.root = None
        unmerge_node_list = [HuffmanNode(wordid, frequency) for wordid, frequency in
                             wordid_frequency_dict.items()]  # 未合并节点list
        self.huffman = [HuffmanNode(wordid, frequency) for wordid, frequency in
                        wordid_frequency_dict.items()]  # 存储所有的叶子节点和中间节点
        # 构建huffman tree
        self.build_tree(unmerge_node_list)
        # 生成huffman code
        self.generate_huffman_code_and_path()

    def merge_node(self, node1, node2):
        sum_frequency = node1.frequency + node2.frequency
        mid_node_id = len(self.huffman)  # 中间节点的value存中间节点id
        father_node = HuffmanNode(mid_node_id, sum_frequency)
        if node1.frequency >= node2.frequency:
            father_node.left_child = node1
            father_node.right_child = node2
        else:
            father_node.left_child = node2
            father_node.right_child = node1
        self.huffman.append(father_node)
        return father_node

    def build_tree(self, node_list):
        while len(node_list) > 1:
            i1 = 0  # 概率最小的节点
            i2 = 1  # 概率第二小的节点
            if node_list[i2].frequency < node_list[i1].frequency:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        [i1, i2] = [i2, i1]
            father_node = self.merge_node(node_list[i1], node_list[i2])  # 合并最小的两个节点
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, father_node)  # 插入新节点
        self.root = node_list[0]

    def generate_huffman_code_and_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            # 顺着左子树走
            while node.left_child or node.right_child:
                code = node.Huffman_code
                path = node.path
                node.left_child.Huffman_code = code + [1]
                node.right_child.Huffman_code = code + [0]
                node.left_child.path = path + [node.word_id]
                node.right_child.path = path + [node.word_id]
                # 把没走过的右子树加入栈
                stack.append(node.right_child)
                node = node.left_child
            word_id = node.word_id
            word_code = node.Huffman_code
            word_path = node.path
            self.huffman[word_id].Huffman_code = word_code
            self.huffman[word_id].path = word_path
            # 把节点计算得到的霍夫曼码、路径  写入词典的数值中
            self.wordid_code[word_id] = word_code
            self.wordid_path[word_id] = word_path

    # 获取所有词的正向节点id和负向节点id数组
    def get_all_pos_and_neg_path(self):
        positive = []  # 所有词的正向路径数组
        negative = []  # 所有词的负向路径数组
        for word_id in range(self.word_count):
            pos_id = []  # 存放一个词 路径中的正向节点id
            neg_id = []  # 存放一个词 路径中的负向节点id
            for i, code in enumerate(self.huffman[word_id].Huffman_code):
                if code == 1:
                    pos_id.append(self.huffman[word_id].path[i])
                else:
                    neg_id.append(self.huffman[word_id].path[i])
            positive.append(pos_id)
            negative.append(neg_id)
        return positive, negative


class DeepWalk( AbstractClass):
    def __init__(self, args, logger):
        super(DeepWalk, self).__init__(args, logger)

        rw_path = args.random_walk_path if args.random_walk_path else osp.join(args.data_dir, f'random_walks.{args.task}.dat')
        if osp.isfile(rw_path):
            self.print("Reading random walks from cache...")
            pickle_file = open(rw_path, 'rb')
            self.deepwalk_corpus = pickle.load(pickle_file)
            pickle_file.close()
        else:
            self.print(f"Constructing random walks and save to {rw_path}...")
            self.deepwalk_corpus = self.build_deepwalk_corpus(args.num_walker, args.path_length, args.alpha)
            pickle_file = open(rw_path, 'wb')
            pickle.dump(self.deepwalk_corpus, pickle_file)
            pickle_file.close()

        self.pre_train_embedding = utils.init_embed(self.num_node*2-1, self.embed_dim)
        self.word_freq = self.init_word_freq_dict(self.deepwalk_corpus)
        self.huffman_tree = HuffmanTree(self.word_freq)
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        self.word_count = len(self.huffman_pos_path)
        self.word_pairs_queue = deque()

        self.model = SkipGramModel(self.pre_train_embedding)
        self.model.to(args.device)

    def train(self, args, evaluate_funcs):
        """train the whole graph gan network"""
        optim_map = {'Adam': optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta,
                     'SGD': functools.partial(optim.SGD, momentum=0.9)}
        if args.lr > 0:
            optimizer = optim_map[args.optimizer](filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=args.lr)
        else:
            optimizer = optim_map[args.optimizer](filter(lambda p: p.requires_grad, self.model.parameters()))

        # evaluate pre-train embed
        self.evaluate(args, evaluate_funcs)

        epoch_count = self.num_node * 10
        num_data = 0
        losses = []
        train_start_time = time.time()
        for epoch, (pairs, labels) in self.get_train_data(self.deepwalk_corpus, args.epochs, epoch_count, args.dw_window_size):
            num_data += len(labels)
            for batch_data in tqdm(self.get_batch_data_for_model(pairs, labels, args.bs)):
                optimizer.zero_grad()
                loss = self.model(*batch_data)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss = np.mean(losses)
                self.print(f'Epoch: {epoch:04d} loss: {avr_loss:.4f} data:{num_data} duration: {duration:.2f}')
                self.stats['graph_loss'].append((epoch, avr_loss))
                losses = []
                num_data = 0

            if epoch % args.save_every == 0:
                flag = self.evaluate(args, evaluate_funcs, epoch, optimizer)
                if args.early_stop and flag:
                    break

        self.save_all(args)

    def init_word_freq_dict(self, corpus):
        word_freq = defaultdict(int)
        for path in tqdm(corpus, desc='Building frequency ditionary..'):
            # self.word_count_sum += len(path)
            for word in path:
                word_freq[word] += 1
        return dict(word_freq)

    def random_walk(self, path_length, alpha=0, start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [random.choice(self.root_nodes)]

        graph = self.graph
        while len(path) < path_length:
            cur = path[-1]
            if len(graph[cur]) > 0:
                if random.random() >= alpha:
                    path.append(random.choice(graph[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [node for node in path]

    def build_deepwalk_corpus(self, num_walkers, path_length, alpha=0):
        walks = []
        nodes = self.root_nodes.copy()
        for cnt in trange(num_walkers, desc='Random Walking...'):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(path_length, alpha=alpha, start=node))
        return walks

    def generate_window_pairs(self, path, window_size):
        pos_pairs = []
        for i, center_node in enumerate(path):
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(path))):
                if i != j:
                    pos_pairs.append((center_node, path[j]))
        return pos_pairs

    def get_train_data(self, corpus, n_epoch, count, window_size):
        random.shuffle(corpus)
        labels = []
        pairs = []
        cnt = 0
        flag = 0
        while cnt < n_epoch:
            for path in corpus:
                if flag:
                    labels = []
                    pairs = []
                    flag = 0
                pos_pairs = self.generate_window_pairs(path, window_size)
                for pair in pos_pairs:
                    if pair[1] >= self.word_count:
                        continue
                    n_pos = len(self.huffman_pos_path[pair[1]])
                    pairs += zip([pair[0]] * n_pos, self.huffman_pos_path[pair[1]])
                    n_neg = len(self.huffman_neg_path[pair[1]])
                    pairs += zip([pair[0]] * n_neg, self.huffman_neg_path[pair[1]])
                    labels += [1]*n_pos+[0]*n_neg
                if len(labels) >= count:
                    labels = np.array(labels)
                    pairs = np.array(pairs)
                    indices = np.arange(len(labels))
                    np.random.shuffle(indices)
                    labels = labels[indices]
                    pairs = pairs[indices]
                    cnt += 1
                    flag = 1
                    yield cnt, (pairs, labels)

    def get_batch_data_for_model(self, pairs, labels, batch_size, convert=True, total=None):
        if convert:
            for start in range(0, len(labels), batch_size):
                end = start + batch_size
                yield torch.LongTensor(pairs[start:end]).to(self.device), torch.DoubleTensor(labels[start:end]).to(self.device)
        else:
            num = len(labels)
            for start in range(0, total, batch_size):
                start %= num
                end = start + batch_size
                yield pairs[start:end], labels[start:end]

