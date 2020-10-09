import os
import os.path as osp
import json, pickle
import time
import random
import numpy as np
import functools
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import src.utils as utils
from src.abstract_model import TransformTaxoSkipGramModel, AbstractTaxoPTE


class TaxoTransformPTE(AbstractTaxoPTE):
    def __init__(self, args, logger):
        super(TaxoTransformPTE, self).__init__(args, logger)
        if args.rand_init:
            self.embed_dim = args.embed_dim
            self.pre_train_embedding = utils.init_embed(self.num_node, self.embed_dim)
            taxo_init_embed = utils.init_embed(self.num_category+1, self.embed_dim)
        else:
            taxo_init_embed = self.init_taxo_embed(self.pre_train_embedding)

        max_level = 0
        for _, paths in self.nodeid2path.items():
            for p in paths:
                max_level = max(max_level, len(p))
        self.max_level = max_level

        self.sgm = TransformTaxoSkipGramModel(self.pre_train_embedding, taxo_init_embed, max_level, args.top_down, args.stacked_transform)
        self.sgm.to(args.device)

    def train(self, args, evaluate_func):
        """train the whole graph gan network"""
        log_dir = args.log_dir

        optim_map = {'Adam': optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9)}
        if args.lr > 0:
            optimizer = optim_map[args.optimizer](filter(lambda p: p.requires_grad, self.sgm.parameters()), lr=args.lr)
        else:
            optimizer = optim_map[args.optimizer](filter(lambda p: p.requires_grad, self.sgm.parameters()))

        # evaluate pre-train embed
        if args.task == 'taxonomy':
            acc, std = evaluate_func(self)
        else:
            acc, std = evaluate_func(self.sgm.get_embed())
        self.print(f'[INIT EVALUATION] acc={acc:.2f} += {std:.2f}')

        batch_count = args.bs * (1 + args.negative_sample_size)
        batch_range = (self.num_node * 10) // batch_count
        best_acc, data_num, taxo_data_num = -1, 0, 0
        graph_loss, taxo_loss = [], []
        stats = {'graph_loss': [], 'acc': [], 'taxo_loss': []}
        train_start_time = time.time()
        for epoch in range(args.epochs):
            for _ in trange(batch_range):
                data_num += batch_count
                optimizer.zero_grad()
                batch_data = self.sample_data(args.bs, args.negative_sample_size)
                loss = self.sgm(*batch_data)
                loss.backward()
                graph_loss.append(loss.item())
                optimizer.step()

            for _ in trange(batch_range):
                optimizer.zero_grad()
                batch_data = self.advanced_sample_taxo(args.bs, args.negative_sample_size)
                taxo_data_num += len(batch_data[1])
                loss = args.lambda_taxo * self.sgm.forward_taxo_transform(*batch_data)
                assert not torch.isnan(loss)
                loss.backward()
                taxo_loss.append(loss.item())
                optimizer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                graph_avr_loss = np.mean(graph_loss)
                taxo_avr_loss = np.mean(taxo_loss)
                self.print(
                    f'Epoch: {epoch:04d} graph loss: {graph_avr_loss:.4f} graph data:{data_num:d} taxo loss: {taxo_avr_loss:.4f} taxo data:{taxo_data_num:d} duration: {duration:.2f}')
                stats['graph_loss'].append((epoch, graph_avr_loss))
                stats['taxo_loss'].append((epoch, taxo_avr_loss))
                graph_loss, taxo_loss = [], []
                data_num, taxo_data_num = 0, 0

            if epoch % args.save_every == 0:
                embed = self.sgm.get_embed()
                if args.task == 'taxonomy':
                    acc, std = evaluate_func(self)
                else:
                    acc, std = evaluate_func(embed)
                stats['acc'].append((epoch, acc, std))

                if acc > best_acc:
                    best_acc = acc
                    best_std = std
                    best_epoch = epoch
                    best_model = self.sgm.state_dict()
                    best_opt = optimizer.state_dict()
                    self.save_embedding(args.embed_path_d, embed)
                    count = 0
                else:
                    if args.early_stop:
                        count += args.save_every
                    if count >= args.patience:
                        self.print('early stopped!')
                        break

                self.print(f'[EVALUATION] acc={acc:.2f} += {std:.2f}')

        json.dump(stats, open(osp.join(log_dir, 'stats.json'), 'w'), indent=4)
        self.save_checkpoint({
            'args': args,
            'model': best_model,
            'optimizer': best_opt
        }, log_dir, f'epoch{best_epoch}_acc{best_acc:.2f}.pth.tar', True)
        self.print(f'best acc g ={best_acc:.2f} +- {best_std:.2f} @epoch:{best_epoch:d}')



