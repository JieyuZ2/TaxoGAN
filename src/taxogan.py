import math
import random
import time
import collections
import numpy as np
from tqdm import tqdm, trange

from src import utils
from src.utils import l2_loss, softmax
from src.abstract_model import AbstractGAN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.enabled = True


class AbstractGANClass(nn.Module):
    def __init__(self, lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform=True):
        super(AbstractGANClass, self).__init__()
        self.lamb = lamb
        self.transform = transform
        self.lambda_taxo = lambda_taxo
        self.n_node, self.emd_size = node_emd_init.shape
        self.n_category = len(taxo_emd_init)

        if transform:
            self.max_level = max_level
            self.transforms = torch.DoubleTensor(max_level, self.emd_size, self.emd_size)
            stdv = 1. / math.sqrt(self.emd_size)
            self.transforms.data.uniform_(-stdv, stdv)
            self.transforms = nn.Parameter(self.transforms)

        self.node_emd = nn.Embedding(self.n_node, self.emd_size, max_norm=1)
        self.bias_vector = nn.Parameter(torch.zeros(self.n_node))
        # self.bias_vector = torch.zeros(self.n_node).double().cuda()
        self.taxo_emd = nn.Embedding(self.n_category, self.emd_size, max_norm=1)
        self.taxo_bias_vector = nn.Parameter(torch.zeros(self.n_category))

        self.double()

    def forward(self, node_ids, neighbor_ids, reward):
        pass

    def forward_taxo(self, nodes, labels, levels):
        if self.transform:
            transforms = F.normalize(self.transforms, p=2, dim=1)
            # transforms = self.transforms
            node_embedding = self.node_emd(nodes[:, 0]).unsqueeze(1) @ transforms[levels]
        else:
            node_embedding = self.node_emd(nodes[:, 0])
        category_embedding = self.taxo_emd(nodes[:, 1])
        bias = self.taxo_bias_vector.gather(0, nodes[:, 1])
        score = (node_embedding.squeeze(1) * category_embedding).sum(dim=1) + bias
        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss * self.lambda_taxo

    def get_embed(self):
        return self.node_emd.weight.data.cpu().numpy()

    def get_taxo_embed(self):
        return self.taxo_emd.weight.data.cpu().numpy()

    def get_taxo_bias(self):
        return self.taxo_bias_vector.data.cpu().numpy()

    def get_transforms(self):
        with torch.no_grad():
            if self.transform:
                transforms = F.normalize(self.transforms.data, p=2, dim=1)
                # transforms = self.transforms.data
                return transforms
            else:
                return None

    def return_node_embedding_by_path(self, node, transforms, level):
        with torch.no_grad():
            if self.transform:
                node_embed = self.node_emd.weight.data[node] @ transforms[:level]
            else:
                node_embed = self.node_emd.weight.data[node].repeat(level, 1)
            return node_embed.data.cpu().numpy()

    def set_lambda_taxo(self, l):
        self.lambda_taxo = l


class Generator(AbstractGANClass):
    def __init__(self, lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform):
        super(Generator, self).__init__(lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform)

    def forward(self, node_ids, neighbor_ids, reward):
        node_embedding = self.node_emd(node_ids)
        neighbor_node_embedding = self.node_emd(neighbor_ids)
        bias = self.bias_vector.gather(0, neighbor_ids)
        score = (node_embedding * neighbor_node_embedding).sum(dim=1) + bias
        prob = score.sigmoid().clamp(1e-5, 1)
        loss = -(prob.log() * reward).mean() + self.lamb * (l2_loss(bias))
        return loss

    def get_all_score(self):
        node_emd = self.node_emd.weight.data.cpu().numpy()
        score = node_emd.dot(node_emd.T) + self.bias_vector.data.cpu().numpy()
        return score


class Discriminator(AbstractGANClass):
    def __init__(self, lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform):
        super(Discriminator, self).__init__(lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform)

    def forward(self, node_ids, neighbor_ids, label):
        node_embedding = self.node_emd(node_ids)
        neighbor_node_embedding = self.node_emd(neighbor_ids)
        bias = self.bias_vector.gather(0, neighbor_ids)
        score = (node_embedding * neighbor_node_embedding).sum(dim=1) + bias
        loss = F.binary_cross_entropy_with_logits(score, label) + self.lamb * (l2_loss(bias))
        return loss

    def get_reward(self, node_ids, neighbor_ids):
        with torch.no_grad():
            node_embedding = self.node_emd(node_ids)
            neighbor_node_embedding = self.node_emd(neighbor_ids)
            bias = self.bias_vector.gather(0, neighbor_ids)
            score = (node_embedding * neighbor_node_embedding).sum(dim=1) + bias
            reward = (score.data.clamp(-10, 10).exp() + 1).log()
            return reward.data


class Generator_V2(Generator):
    def __init__(self, lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform):
        super(Generator_V2, self).__init__(lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform)

    def forward_taxo(self, node_ids, category_ids, reward, levels):
        if self.transform:
            transforms = F.normalize(self.transforms, p=2, dim=1)
            # transforms = self.transforms
            node_embedding = self.node_emd(node_ids).unsqueeze(1) @ transforms[levels]
        else:
            node_embedding = self.node_emd(node_ids)
        category_embedding = self.taxo_emd(category_ids)
        bias = self.taxo_bias_vector.gather(0, category_ids)
        score = (node_embedding.squeeze(1) * category_embedding).sum(dim=1) + bias
        prob = score.sigmoid().clamp(1e-5, 1)
        loss = -(prob.log() * reward).mean() + self.lamb * (l2_loss(bias))
        return loss * self.lambda_taxo

    def get_taxo_all_score(self, levels):
        with torch.no_grad():
            if self.transform:
                transforms = F.normalize(self.transforms.data, p=2, dim=1)
                # transforms = self.transforms.data
                node_emd = self.node_emd.weight.data @ transforms
                cols = []
                for taxo, level in zip(self.taxo_emd.weight.data, levels):
                    cols.append(node_emd[level] @ taxo.view(-1, 1))
                score = (torch.cat(cols, dim=1) + self.taxo_bias_vector.data).cpu().numpy()
            else:
                node_emd = self.node_emd.weight.data.cpu().numpy()
                taxo_emd = self.taxo_emd.weight.data.cpu().numpy()
                score = node_emd.dot(taxo_emd.T) + self.taxo_bias_vector.data.cpu().numpy()
            return score


class Discriminator_V2(Discriminator):
    def __init__(self, lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform):
        super(Discriminator_V2, self).__init__(lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform)

    def forward_taxo(self, node_ids, category_ids, label, levels):
        if self.transform:
            transforms = F.normalize(self.transforms, p=2, dim=1)
            # transforms = self.transforms
            node_embedding = self.node_emd(node_ids).unsqueeze(1) @ transforms[levels]
        else:
            node_embedding = self.node_emd(node_ids)
        category_embedding = self.taxo_emd(category_ids)
        bias = self.taxo_bias_vector.gather(0, category_ids)
        score = (node_embedding.squeeze(1) * category_embedding).sum(dim=1) + bias
        loss = F.binary_cross_entropy_with_logits(score, label) + self.lamb * (l2_loss(bias))
        return loss * self.lambda_taxo

    def get_reward_taxo(self, node_ids, category_ids, levels):
        with torch.no_grad():
            if self.transform:
                transforms = F.normalize(self.transforms, p=2, dim=1)
                # transforms = self.transforms
                node_embedding = self.node_emd(node_ids).unsqueeze(1) @ transforms[levels]
            else:
                node_embedding = self.node_emd(node_ids)
            category_embedding = self.taxo_emd(category_ids)
            bias = self.taxo_bias_vector.gather(0, category_ids)
            score = (node_embedding.squeeze(1) * category_embedding).sum(dim=1) + bias
            reward = (score.data.clamp(-10, 10).exp() + 1).log()
            return reward.data


class Generator_V3(Generator_V2):
    def __init__(self, lamb, node_emd_init, taxo_emd_init, lambda_taxo, n_category, transform):
        super(Generator_V3, self).__init__(lamb, node_emd_init, taxo_emd_init, lambda_taxo, n_category, transform)

    def get_taxo_all_score(self):
        with torch.no_grad():
            node_emd = self.node_emd.weight.data.cpu().numpy()
            if self.transform:
                transforms = F.normalize(self.transforms.data, p=2, dim=1)
                # transforms = self.transforms.data
                taxo_emd = (self.taxo_emd.weight.data.unsqueeze(1) @ transforms.transpose(2, 1)).squeeze(1).cpu().numpy()
            else:
                taxo_emd = self.taxo_emd.weight.data.cpu().numpy()
            score = node_emd.dot(taxo_emd.T) + self.taxo_bias_vector.data.cpu().numpy()
            return score

    def return_node_embedding_by_path(self, node, transforms, parents):
        with torch.no_grad():
            if self.transform:
                node_embed = self.node_emd.weight.data[node] @ transforms[parents]
            else:
                node_embed = self.node_emd.weight.data[node].repeat(len(parents), 1)
            return node_embed.data.cpu().numpy()


class Discriminator_V3(Discriminator_V2):
    def __init__(self, lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform):
        super(Discriminator_V3, self).__init__(lamb, node_emd_init, taxo_emd_init, lambda_taxo, max_level, transform)

    def return_node_embedding_by_path(self, node, transforms, parents):
        with torch.no_grad():
            if self.transform:
                node_embed = self.node_emd.weight.data[node] @ transforms[parents]
            else:
                node_embed = self.node_emd.weight.data[node].repeat(len(parents), 1)
            return node_embed.data.cpu().numpy()


class TaxoGAN_V1(AbstractGAN):
    def __init__(self, args, logger=None):
        super(TaxoGAN_V1, self).__init__(args, logger)

        self.taxo_parent2children, self.taxo_child2parents, self.nodeid2category, self.category2nodeid, self.category2id, self.nodeid2path \
            = utils.read_taxos(args.taxo_file, args.taxo_assign_file)
        self.root_nodes = [i for i in self.root_nodes if i in self.nodeid2category]
        categories = list(self.category2nodeid.keys())
        self.nodeid2neg_category = {node: [i for i in categories if i not in self.nodeid2category[node]] for node in
                                    self.root_nodes}

        edgedistdict, taxodistdict = self.make_dist()
        self.taxo_edge_alias_sampler = utils.VoseAlias(edgedistdict)
        self.taxo_alias_sampler = utils.VoseAlias(taxodistdict)
        self.num_category = len(self.category2id)
        self.taxo_rootid = self.category2id['root']
        self.category2level = utils.category2level(range(self.num_category), self.taxo_child2parents)

        self.print(f"num of category: {self.num_category}")

        if args.rand_init:
            taxo_init_embed = utils.init_embed(self.num_category, self.embed_dim)
        else:
            taxo_init_embed = self.init_taxo_embed(self.pre_train_embedding)

        max_level = 0
        for _, paths in self.nodeid2path.items():
            for p in paths:
                max_level = max(max_level, len(p))
        self.max_level = max_level

        # build the model
        self.generator = Generator(args.lambda_g, self.pre_train_embedding, taxo_init_embed, args.lambda_taxo, max_level+1, args.transform)
        self.discriminator = Discriminator(args.lambda_d, self.pre_train_embedding, taxo_init_embed, args.lambda_taxo, max_level+1, args.transform)
        self.generator.to(args.device)
        self.discriminator.to(args.device)

    def train(self, args, evaluate_funcs):
        """train the whole graph gan network"""
        g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=args.lr_g)
        d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=args.lr_d)

        # evaluate pre-train embed
        self.evaluate(args, evaluate_funcs)

        epoch_count = self.num_node * 10
        batch_count = args.bs * (1 + args.negative_sample_size)
        d_t_loss, g_t_loss = [], []
        d_t_data, g_t_data = 0, 0
        d_g_loss, g_g_loss = [], []
        d_g_data, g_g_data = 0, 0
        train_start_time = time.time()
        self.update_all_score()
        for epoch in range(args.epochs):

            for d_epoch in trange(args.epochs_d):
                center_nodes, neighbor_nodes, labels = self.sample_for_d(epoch_count, args.n_sample_d, args.update_ratio)
                d_g_data += len(labels)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, labels, args.bs_d):
                    d_optimizer.zero_grad()
                    loss = self.discriminator(*batch_data)
                    d_g_loss.append(loss.item())
                    loss.backward()
                    d_optimizer.step()
            for g_epoch in trange(args.epochs_g):
                center_nodes, neighbor_nodes = self.sample_for_g(epoch_count, args.n_sample_g, args.gan_window_size, args.update_ratio)
                center_nodes = torch.LongTensor(center_nodes).to(self.device)
                neighbor_nodes = torch.LongTensor(neighbor_nodes).to(self.device)
                rewards = self.discriminator.get_reward(center_nodes, neighbor_nodes)
                g_g_data += len(rewards)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, rewards, args.bs_g, False):
                    g_optimizer.zero_grad()
                    loss = self.generator(*batch_data)
                    g_g_loss.append(loss.item())
                    loss.backward()
                    g_optimizer.step()
                self.update_all_score()

            batch_range = d_g_data // batch_count
            for _ in trange(batch_range):
                d_optimizer.zero_grad()
                batch_data = self.sample_taxo(args.bs, args.negative_sample_size)
                d_t_data += len(batch_data[1])
                loss = self.discriminator.forward_taxo(*batch_data)
                loss.backward()
                d_t_loss.append(loss.item())
                d_optimizer.step()
            batch_range = g_g_data // batch_count
            for _ in trange(batch_range):
                g_optimizer.zero_grad()
                batch_data = self.sample_taxo(args.bs, args.negative_sample_size)
                g_t_data += len(batch_data[1])
                loss = self.generator.forward_taxo(*batch_data)
                loss.backward()
                g_t_loss.append(loss.item())
                g_optimizer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_g_g, avr_loss_d_g = np.mean(g_g_loss), np.mean(d_g_loss)
                avr_loss_g_t, avr_loss_d_t = np.mean(g_t_loss), np.mean(d_t_loss)
                self.print(
                    f'Epoch: {epoch:04d} graph: d_loss: {avr_loss_d_g:.4f} d_data:{d_g_data} g_loss: {avr_loss_g_g:.4f} g_data:{g_g_data} duration: {duration:.2f}')
                self.print(
                    f'Epoch: {epoch:04d} taxonomy: d_loss: {avr_loss_d_t:.4f} d_data:{d_t_data} g_loss: {avr_loss_g_t:.4f} g_data:{g_t_data} duration: {duration:.2f}')
                self.stats['d_g_loss'].append((epoch, avr_loss_d_g))
                self.stats['g_g_loss'].append((epoch, avr_loss_g_g))
                self.stats['d_t_loss'].append((epoch, avr_loss_d_t))
                self.stats['g_t_loss'].append((epoch, avr_loss_g_t))
                d_t_loss, g_t_loss = [], []
                d_t_data, g_t_data = 0, 0
                d_g_loss, g_g_loss = [], []
                d_g_data, g_g_data = 0, 0

            if epoch % args.save_every == 0:
                flag = self.evaluate(args, evaluate_funcs, epoch, d_optimizer, g_optimizer)
                if args.early_stop and flag:
                    break

        self.save_all(args)

    def sample_taxo(self, batch_size, negsample_size):
        sampled_pairs = []
        levels = []
        labels = []
        for (node, category) in self.taxo_edge_alias_sampler.sample_n(batch_size):
            parent = self.taxo_child2parents[category]
            true_category = self.nodeid2category[node]
            siblings = [i for i in self.taxo_parent2children[parent] if i not in true_category]
            if len(siblings):
                sampled_pairs.append((node, category))
                level = self.category2level[category]
                levels.extend([level]*(1+negsample_size))
                labels.extend([1] + [0] * negsample_size)
                for sampledcategory in self.taxo_alias_sampler.sample_from(siblings, negsample_size):
                    sampled_pairs.append((node, sampledcategory))
        return torch.LongTensor(sampled_pairs).to(self.device), torch.DoubleTensor(labels).to(self.device), torch.LongTensor(levels).to(self.device)


class TaxoGAN_V2(AbstractGAN):
    def __init__(self, args, logger=None):
        super(TaxoGAN_V2, self).__init__(args, logger)

        self.taxo_parent2children, self.taxo_child2parents, self.nodeid2category, self.category2nodeid, self.category2id, self.nodeid2path \
            = utils.read_taxos(args.taxo_file, args.taxo_assign_file)
        self.root_nodes = [i for i in self.root_nodes if i in self.nodeid2category]

        self.num_category = len(self.category2id)
        self.taxo_rootid = self.category2id['root']
        self.category2level = utils.category2level(range(self.num_category), self.taxo_child2parents)
        self.levels = [self.category2level[i] for i in range(self.num_category)]
        self.level2category = collections.defaultdict(list)
        self.max_level = max(self.levels)
        for i, level in enumerate(self.levels):
            self.level2category[level].append(i)

        self.print(f"num of category: {self.num_category}")

        if args.rand_init:
            taxo_init_embed = utils.init_embed(self.num_category, self.embed_dim)
        else:
            taxo_init_embed = self.init_taxo_embed(self.pre_train_embedding)


        # build the model
        self.generator = Generator_V2(args.lambda_g, self.pre_train_embedding, taxo_init_embed, args.lambda_taxo, self.max_level+1, args.transform)
        self.discriminator = Discriminator_V2(args.lambda_d, self.pre_train_embedding, taxo_init_embed, args.lambda_taxo, self.max_level+1, args.transform)
        self.generator.to(args.device)
        self.discriminator.to(args.device)

        self.taxo_all_score = None

    def train(self, args, evaluate_funcs):
        """train the whole graph gan network"""

        g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=args.lr_g)
        d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=args.lr_d)

        # evaluate pre-train embed
        self.evaluate(args, evaluate_funcs)

        epoch_count = 5 * self.num_node
        d_t_loss, g_t_loss = [], []
        d_t_data, g_t_data = 0, 0
        d_g_loss, g_g_loss = [], []
        d_g_data, g_g_data = 0, 0
        train_start_time = time.time()
        self.update_all_score()
        self.update_taxo_all_score()
        for epoch in range(args.epochs):

            for d_epoch in trange(args.epochs_d):
                center_nodes, neighbor_nodes, labels = self.sample_for_d(epoch_count, args.n_sample_d, args.update_ratio)
                d_g_data += len(labels)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, labels, args.bs_d):
                    d_optimizer.zero_grad()
                    loss = self.discriminator(*batch_data)
                    d_g_loss.append(loss.item())
                    loss.backward()
                    d_optimizer.step()
            for g_epoch in trange(args.epochs_g):
                center_nodes, neighbor_nodes = self.sample_for_g(epoch_count, args.n_sample_g, args.gan_window_size, args.update_ratio)
                center_nodes = torch.LongTensor(center_nodes).to(self.device)
                neighbor_nodes = torch.LongTensor(neighbor_nodes).to(self.device)
                rewards = self.discriminator.get_reward(center_nodes, neighbor_nodes)
                g_g_data += len(rewards)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, rewards, args.bs_g, False):
                    g_optimizer.zero_grad()
                    loss = self.generator(*batch_data)
                    g_g_loss.append(loss.item())
                    loss.backward()
                    g_optimizer.step()
                self.update_all_score()

            for d_epoch in trange(args.epochs_d):
                nodes, categories, labels, levels = self.sample_taxo(d_g_data, args.update_ratio, True)
                d_t_data += len(labels)
                for batch_data in self.get_batch_data_w_levels(nodes, categories, labels, levels, args.bs_d):
                    d_optimizer.zero_grad()
                    loss = self.discriminator.forward_taxo(*batch_data)
                    d_t_loss.append(loss.item())
                    loss.backward()
                    d_optimizer.step()
            for g_epoch in trange(args.epochs_g):
                nodes, categories, levels = self.sample_taxo(g_g_data/args.epochs_g, args.update_ratio, False)
                nodes = torch.LongTensor(nodes).to(self.device)
                categories = torch.LongTensor(categories).to(self.device)
                g_t_data += len(levels)
                for (nodes, categories, levels) in self.get_batch_data(nodes, categories, levels, args.bs_g, False):
                    rewards = self.discriminator.get_reward_taxo(nodes, categories, levels)
                    g_optimizer.zero_grad()
                    loss = self.generator.forward_taxo(nodes, categories, rewards, levels)
                    g_t_loss.append(loss.item())
                    loss.backward()
                    g_optimizer.step()
                self.update_taxo_all_score()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_g_g, avr_loss_d_g = np.mean(g_g_loss), np.mean(d_g_loss)
                avr_loss_g_t, avr_loss_d_t = np.mean(g_t_loss), np.mean(d_t_loss)
                self.print(
                    f'Epoch: {epoch:04d} graph: d_loss: {avr_loss_d_g:.4f} d_data:{d_g_data} g_loss: {avr_loss_g_g:.4f} g_data:{g_g_data} duration: {duration:.2f}')
                self.print(
                    f'Epoch: {epoch:04d} taxonomy: d_loss: {avr_loss_d_t:.4f} d_data:{d_t_data} g_loss: {avr_loss_g_t:.4f} g_data:{g_t_data} duration: {duration:.2f}')
                self.stats['d_g_loss'].append((epoch, avr_loss_d_g))
                self.stats['g_g_loss'].append((epoch, avr_loss_g_g))
                self.stats['d_t_loss'].append((epoch, avr_loss_d_t))
                self.stats['g_t_loss'].append((epoch, avr_loss_g_t))
                d_t_loss, g_t_loss = [], []
                d_t_data, g_t_data = 0, 0
                d_g_loss, g_g_loss = [], []
                d_g_data, g_g_data = 0, 0

            if epoch % args.save_every == 0:
                if epoch % args.save_every == 0:
                    flag = self.evaluate(args, evaluate_funcs, epoch, d_optimizer, g_optimizer)
                    if args.early_stop and flag:
                        break

        self.save_all(args)

    def get_batch_data_w_levels(self, left, right, label, levels, batch_size, convert=True):
        if convert:
            for start in range(0, len(label), batch_size):
                end = start + batch_size
                yield torch.LongTensor(left[start:end]).to(self.device), torch.LongTensor(right[start:end]).to(self.device), torch.DoubleTensor(label[start:end]).to(self.device), levels[start:end]
        else:
            for start in range(0, len(label), batch_size):
                end = start + batch_size
                yield left[start:end], right[start:end], label[start:end]

    def update_taxo_all_score(self):
        self.taxo_all_score = self.generator.get_taxo_all_score(self.levels)

    def make_dist(self, power=0.75):
        taxonodedistdict = collections.defaultdict(int)
        taxodistdict = collections.defaultdict(int)
        weightsum = 0
        negprobsum = 0
        # can work with weighted edge, but currently we do not have
        weight = 1
        for node, cs in self.nodeid2category.items():
            for c in cs:
                taxonodedistdict[node] = weight
                taxodistdict[c] += weight
                weightsum += weight
                negprobsum += np.power(weight, power)

        for node, outdegree in taxodistdict.items():
            taxodistdict[node] = np.power(outdegree, power) / negprobsum

        for node, outdegree in taxonodedistdict.items():
            taxonodedistdict[node] = np.power(outdegree, power) / negprobsum

        return taxonodedistdict, taxodistdict

    def sample_taxo(self, data_size, update_ratio, sample_for_dis: bool):
        taxo_all_score = self.taxo_all_score
        if sample_for_dis:
            nodes = []
            cates = []
            labels = []
            while len(nodes) < data_size:
                node = random.choice(self.root_nodes)
                paths = self.nodeid2path.get(node, None)
                if paths is None:
                    continue
                true_category = self.nodeid2category[node]
                true_path = random.choice(paths)
                p = self.taxo_rootid
                fake_path = []
                for c in true_path:
                    siblings = [i for i in self.taxo_parent2children[p] if i not in true_category]
                    if not siblings:
                        true_path = true_path[:len(fake_path)]
                        break
                    prob = softmax(taxo_all_score[node, siblings])
                    category_select = np.random.choice(siblings, p=prob)
                    fake_path.append(category_select)
                    p = c
                n_pos, n_neg = len(true_path), len(fake_path)
                cates.extend(true_path)
                labels.extend([1]*n_pos)
                cates.extend(fake_path)
                labels.extend([0]*n_neg)
                nodes.extend([node]*(n_pos+n_neg))
            levels = [self.category2level[c] for c in cates]
            return nodes, cates, labels, levels
        else:
            nodes = []
            cates = []
            while len(nodes)< data_size:
                node = random.choice(self.root_nodes)
                paths = self.nodeid2path.get(node, None)
                if paths is None:
                    continue
                true_category = self.nodeid2category[node]
                true_path = random.choice(paths)
                p = self.taxo_rootid
                for c in true_path:
                    siblings =[i for i in self.taxo_parent2children[p] if i not in true_category]
                    if not siblings:
                        break
                    if len(siblings)>1:
                        prob = softmax(taxo_all_score[node, siblings])
                        siblings = np.random.choice(siblings, size=1, p=prob).tolist()
                    siblings.append(c)
                    cates.extend(siblings)
                    nodes.extend([node]*len(siblings))
                    p = c
            levels = [self.category2level[c] for c in cates]
            return nodes, cates, levels



class TaxoGAN_V3(AbstractGAN):
    def __init__(self, args, logger=None):
        super(TaxoGAN_V3, self).__init__(args, logger)

        self.taxo_parent2children, self.taxo_child2parents, self.nodeid2category, self.category2nodeid, self.category2id, self.nodeid2path \
            = utils.read_taxos(args.taxo_file, args.taxo_assign_file)
        self.root_nodes = [i for i in self.root_nodes if i in self.nodeid2category]

        self.num_category = len(self.category2id)
        self.taxo_rootid = self.category2id['root']

        self.print(f"num of category: {self.num_category}")

        if args.rand_init:
            taxo_init_embed = utils.init_embed(self.num_category, self.embed_dim)
        else:
            taxo_init_embed = self.init_taxo_embed(self.pre_train_embedding)

        # build the model
        self.generator = Generator_V3(args.lambda_g, self.pre_train_embedding, taxo_init_embed, args.lambda_taxo, self.num_category, args.transform)
        self.discriminator = Discriminator_V3(args.lambda_d, self.pre_train_embedding, taxo_init_embed, args.lambda_taxo, self.num_category, args.transform)
        self.generator.to(args.device)
        self.discriminator.to(args.device)

        self.taxo_all_score = None

    def train(self, args, evaluate_funcs):
        """train the whole graph gan network"""

        g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=args.lr_g)
        d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=args.lr_d)

        # evaluate pre-train embed
        self.evaluate(args, evaluate_funcs)

        epoch_count = 5 * self.num_node
        epoch_count_taxo = 5 * len(self.root_nodes)
        d_t_loss, g_t_loss = [], []
        d_t_data, g_t_data = 0, 0
        d_g_loss, g_g_loss = [], []
        d_g_data, g_g_data = 0, 0
        train_start_time = time.time()
        self.update_all_score()
        self.update_taxo_all_score()
        for epoch in range(args.epochs):

            for d_epoch in trange(args.epochs_d):
                center_nodes, neighbor_nodes, labels = self.sample_for_d(epoch_count, args.n_sample_d, args.update_ratio)
                d_g_data += len(labels)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, labels, args.bs_d):
                    d_optimizer.zero_grad()
                    loss = self.discriminator(*batch_data)
                    d_g_loss.append(loss.item())
                    loss.backward()
                    d_optimizer.step()
            for g_epoch in trange(args.epochs_g):
                center_nodes, neighbor_nodes = self.sample_for_g(epoch_count, args.n_sample_g, args.gan_window_size, args.update_ratio)
                center_nodes = torch.LongTensor(center_nodes).to(self.device)
                neighbor_nodes = torch.LongTensor(neighbor_nodes).to(self.device)
                rewards = self.discriminator.get_reward(center_nodes, neighbor_nodes)
                g_g_data += len(rewards)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, rewards, args.bs_g, False):
                    g_optimizer.zero_grad()
                    loss = self.generator(*batch_data)
                    g_g_loss.append(loss.item())
                    loss.backward()
                    g_optimizer.step()
                self.update_all_score()

            for d_epoch in trange(args.epochs_d):
                nodes, categories, labels, parents = self.sample_taxo(epoch_count_taxo/args.epochs_d, args.update_ratio, True)
                d_t_data += len(labels)
                for batch_data in self.get_batch_data_w_levels(nodes, categories, labels, parents, args.bs_d):
                    d_optimizer.zero_grad()
                    loss = self.discriminator.forward_taxo(*batch_data)
                    d_t_loss.append(loss.item())
                    loss.backward()
                    d_optimizer.step()
            for g_epoch in trange(args.epochs_g):
                nodes, categories, parents = self.sample_taxo(epoch_count_taxo/args.epochs_g, args.update_ratio, False)
                nodes = torch.LongTensor(nodes).to(self.device)
                categories = torch.LongTensor(categories).to(self.device)
                g_t_data += len(parents)
                for (nodes, categories, parents) in self.get_batch_data(nodes, categories, parents, args.bs_g, False):
                    rewards = self.discriminator.get_reward_taxo(nodes, categories, parents)
                    g_optimizer.zero_grad()
                    loss = self.generator.forward_taxo(nodes, categories, rewards, parents)
                    g_t_loss.append(loss.item())
                    loss.backward()
                    g_optimizer.step()
                self.update_taxo_all_score()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_g_g, avr_loss_d_g = np.mean(g_g_loss), np.mean(d_g_loss)
                avr_loss_g_t, avr_loss_d_t = np.mean(g_t_loss), np.mean(d_t_loss)
                self.print(
                    f'Epoch: {epoch:04d} graph: d_loss: {avr_loss_d_g:.4f} d_data:{d_g_data} g_loss: {avr_loss_g_g:.4f} g_data:{g_g_data} duration: {duration:.2f}')
                self.print(
                    f'Epoch: {epoch:04d} taxonomy: d_loss: {avr_loss_d_t:.4f} d_data:{d_t_data} g_loss: {avr_loss_g_t:.4f} g_data:{g_t_data} duration: {duration:.2f}')
                self.stats['d_g_loss'].append((epoch, avr_loss_d_g))
                self.stats['g_g_loss'].append((epoch, avr_loss_g_g))
                self.stats['d_t_loss'].append((epoch, avr_loss_d_t))
                self.stats['g_t_loss'].append((epoch, avr_loss_g_t))
                d_t_loss, g_t_loss = [], []
                d_t_data, g_t_data = 0, 0
                d_g_loss, g_g_loss = [], []
                d_g_data, g_g_data = 0, 0

            if epoch % args.save_every == 0:
                flag = self.evaluate(args, evaluate_funcs, epoch, d_optimizer, g_optimizer)
                if args.early_stop and flag:
                    break

        self.save_all(args)

    def get_batch_data_w_levels(self, left, right, label, levels, batch_size, convert=True):
        if convert:
            for start in range(0, len(label), batch_size):
                end = start + batch_size
                yield torch.LongTensor(left[start:end]).to(self.device), torch.LongTensor(right[start:end]).to(self.device), torch.DoubleTensor(label[start:end]).to(self.device), levels[start:end]
        else:
            for start in range(0, len(label), batch_size):
                end = start + batch_size
                yield left[start:end], right[start:end], label[start:end]

    def update_taxo_all_score(self):
        self.taxo_all_score = self.generator.get_taxo_all_score()

    def sample_taxo(self, data_size, update_ratio, sample_for_dis: bool):
        taxo_all_score = self.taxo_all_score
        if sample_for_dis:
            nodes = []
            cates = []
            labels = []
            while len(nodes)< data_size:
                node = random.choice(self.root_nodes)
                paths = self.nodeid2path.get(node, None)
                if paths is None:
                    continue
                true_category = self.nodeid2category[node]
                true_path = random.choice(paths)
                p = self.taxo_rootid
                fake_path = []
                for c in true_path:
                    siblings = [i for i in self.taxo_parent2children[p] if i not in true_category]
                    if not siblings:
                        true_path = true_path[:len(fake_path)]
                        break
                    prob = softmax(taxo_all_score[node, siblings])
                    category_select = np.random.choice(siblings, p=prob)
                    fake_path.append(category_select)
                    p = c
                n_pos, n_neg = len(true_path), len(fake_path)
                cates.extend(true_path)
                labels.extend([1]*n_pos)
                cates.extend(fake_path)
                labels.extend([0]*n_neg)
                nodes.extend([node]*(n_pos+n_neg))
            levels = [self.taxo_child2parents[c] for c in cates]
            return nodes, cates, labels, levels
        else:
            nodes = []
            cates = []
            while len(nodes)< data_size:
                node = random.choice(self.root_nodes)
                paths = self.nodeid2path.get(node, None)
                if paths is None:
                    continue
                true_category = self.nodeid2category[node]
                true_path = random.choice(paths)
                p = self.taxo_rootid
                for c in true_path:
                    siblings = [i for i in self.taxo_parent2children[p] if i not in true_category]
                    if not siblings:
                        break
                    if len(siblings)>1:
                        prob = softmax(taxo_all_score[node, siblings])
                        siblings = np.random.choice(siblings, size=1, p=prob).tolist()
                    siblings.append(c)
                    cates.extend(siblings)
                    nodes.extend([node]*len(siblings))
                    p = c
            levels = [self.taxo_child2parents[c] for c in cates]
            return nodes, cates, levels

    def evaluate_taxo(self, test_data, level_by_level=True, split=5):
        correct = []
        taxo_embed = self.eval_model.get_taxo_embed()
        taxo_bias = self.eval_model.get_taxo_bias()
        transforms = self.eval_model.get_transforms()
        for (node, category) in test_data:
            category_path = self.find_path(self.category2id[category])[:-1][::-1]
            parents = [self.taxo_child2parents[i] for i in category_path]
            embeds = self.eval_model.return_node_embedding_by_path(node, transforms, parents)
            p = self.taxo_rootid
            for embed, true_pos in zip(embeds, category_path):
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
