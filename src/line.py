import os
import os.path as osp
import json, pickle
import time
import random
import numpy as np
import functools
from tqdm import tqdm, trange
import torch
import torch.optim as optim

from src import utils
from src.abstract_model import SkipGramModel, AbstractClass


class LINE(AbstractClass):
    def __init__(self, args, logger):
        super(LINE, self).__init__(args, logger)
        self.print(f"Reading data from {args.data_dir}")
        edge_dist_dict, node_dist_dict = utils.makeDist(args.link_file, args.negative_power)
        self.edges_alias_sampler = utils.VoseAlias(edge_dist_dict)
        self.nodes_alias_sampler = utils.VoseAlias(node_dist_dict)

        self.model = SkipGramModel(self.pre_train_embedding)
        self.model.to(args.device)

    def train(self, args, evaluate_funcs):
        """train the whole graph gan network"""
        optim_map = {'Adam': optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta,
                     'SGD': functools.partial(optim.SGD, momentum=0.9)}
        if args.lr > 0:
            optimizer = optim_map[args.optimizer](filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = optim_map[args.optimizer](filter(lambda p: p.requires_grad, self.model.parameters()))

        # evaluate pre-train embed
        self.evaluate(args, evaluate_funcs)

        batch_count = args.bs*(1+args.negative_sample_size)
        batch_range = (self.num_node*10)//batch_count
        data_num = 0
        losses = []
        train_start_time = time.time()
        for epoch in range(args.epochs):
            for _ in trange(batch_range):
                data_num += batch_count
                optimizer.zero_grad()
                batch_data = self.sample_data(args.bs, args.negative_sample_size)
                loss = self.model(*batch_data)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss = np.mean(losses)
                self.print(f'Epoch: {epoch:04d} loss: {avr_loss:.4f} data:{data_num:d} duration: {duration:.2f}')
                self.stats['graph_loss'].append((epoch, avr_loss))
                losses = []
                data_num = 0

            if epoch % args.save_every == 0:
                flag = self.evaluate(args, evaluate_funcs, epoch, optimizer)
                if args.early_stop and flag:
                    break

        self.save_all(args)

    def sample_data(self, batch_size, negsample_size):
        sampled_pairs = []
        labels = ([1] + [0] * negsample_size) * batch_size
        for (src_node, des_node) in self.edges_alias_sampler.sample_n(batch_size):
            if np.random.sample()>0.5:
                src_node, des_node = des_node, src_node
            sampled_pairs.append((src_node, des_node))
            negsample = 0
            while negsample < negsample_size:
                samplednode = self.nodes_alias_sampler.alias_generation()
                if (samplednode == src_node) or (samplednode in self.graph[src_node]):
                    continue
                else:
                    negsample += 1
                    sampled_pairs.append((src_node, samplednode))
        return torch.LongTensor(sampled_pairs).to(self.device), torch.DoubleTensor(labels).to(self.device)


