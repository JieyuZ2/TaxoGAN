import time
import numpy as np
from tqdm import tqdm, trange

from src.utils import l2_loss
from src.abstract_model import AbstractGAN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.enabled = True


class Generator(nn.Module):
    def __init__(self, lambda_gen, node_emd_init):
        super(Generator, self).__init__()
        self.lambda_gen = lambda_gen
        self.n_node, self.emd_size = node_emd_init.shape
        self.node_emd = nn.Embedding.from_pretrained(torch.from_numpy(node_emd_init), freeze=False)
        self.bias_vector = nn.Parameter(torch.zeros(self.n_node))
        self.double()

    def forward(self, node_ids, neighbor_ids, reward):
        node_embedding = self.node_emd(node_ids)
        neighbor_node_embedding = self.node_emd(neighbor_ids)
        bias = self.bias_vector.gather(0, neighbor_ids)
        score = (node_embedding*neighbor_node_embedding).sum(dim=1)+bias
        prob = score.sigmoid().clamp(1e-5, 1)
        loss = -(prob.log()*reward).mean() + self.lambda_gen * (l2_loss(node_embedding)+l2_loss(neighbor_node_embedding)+l2_loss(bias))
        return loss

    def get_all_score(self):
        # with torch.no_grad():
        #     node_emd = self.node_emd.weight.data
        #     score = node_emd.mm(node_emd.t()) + self.bias_vector.data
        #     return score.data.cpu().numpy()
        node_emd = self.node_emd.weight.data.cpu().numpy()
        score = node_emd.dot(node_emd.T) + self.bias_vector.data.cpu().numpy()
        return score

    def get_embed(self):
        return self.node_emd.weight.data.cpu().numpy()


class Discriminator(nn.Module):
    def __init__(self, lambda_dis, node_emd_init):
        super(Discriminator, self).__init__()
        self.lambda_dis = lambda_dis
        self.n_node, self.emd_size = node_emd_init.shape
        self.node_emd = nn.Embedding.from_pretrained(torch.from_numpy(node_emd_init), freeze=False)
        self.bias_vector = nn.Parameter(torch.zeros(self.n_node))
        self.double()

    def forward(self, node_ids, neighbor_ids, label):
        node_embedding = self.node_emd(node_ids)
        neighbor_node_embedding = self.node_emd(neighbor_ids)
        bias = self.bias_vector.gather(0, neighbor_ids)
        score = (node_embedding*neighbor_node_embedding).sum(dim=1)+bias
        loss = (F.binary_cross_entropy_with_logits(score, label)).mean() + \
               self.lambda_dis * (l2_loss(node_embedding)+l2_loss(neighbor_node_embedding)+l2_loss(bias))
        return loss

    def get_reward(self, node_ids, neighbor_ids):
        with torch.no_grad():
            node_embedding = self.node_emd(node_ids)
            neighbor_node_embedding = self.node_emd(neighbor_ids)
            bias = self.bias_vector.gather(0, neighbor_ids)
            score = (node_embedding * neighbor_node_embedding).sum(dim=1) + bias
            reward = (score.data.clamp(-10, 10).exp() + 1).log()
            return reward.data

    def get_embed(self):
        return self.node_emd.weight.data.cpu().numpy()


class GraphGan(AbstractGAN):
    def __init__(self, args, logger=None):
        """initialize the parameters, prepare the data and build the network"""
        super(GraphGan, self).__init__(args, logger)

        # build the model
        self.generator = Generator(args.lambda_g, self.pre_train_embedding)
        self.discriminator = Discriminator(args.lambda_d, self.pre_train_embedding)
        self.generator.to(args.device)
        self.discriminator.to(args.device)

    def train(self, args, evaluate_funcs):
        """train the whole graph gan network"""
        g_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=args.lr_g)
        d_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=args.lr_d)

        # evaluate pre-train embed
        self.evaluate(args, evaluate_funcs)

        epoch_count = self.num_node * 5
        d_loss, g_loss = [], []
        d_data, g_data = 0, 0
        train_start_time = time.time()
        self.update_all_score()
        for epoch in range(args.epochs):
            for d_epoch in trange(args.epochs_d):
                center_nodes, neighbor_nodes, labels = self.sample_for_d(epoch_count, args.n_sample_d, args.update_ratio)
                d_data += len(labels)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, labels, args.bs_d):
                    d_optimizer.zero_grad()
                    loss = self.discriminator(*batch_data)
                    d_loss.append(loss.item())
                    loss.backward()
                    d_optimizer.step()
            for g_epoch in trange(args.epochs_g):
                center_nodes, neighbor_nodes = self.sample_for_g(epoch_count, args.n_sample_g, args.gan_window_size, args.update_ratio)
                center_nodes = torch.LongTensor(center_nodes).to(self.device)
                neighbor_nodes = torch.LongTensor(neighbor_nodes).to(self.device)
                rewards = self.discriminator.get_reward(center_nodes, neighbor_nodes)
                g_data += len(rewards)
                for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, rewards, args.bs_g, False):
                    g_optimizer.zero_grad()
                    loss = self.generator(*batch_data)
                    g_loss.append(loss.item())
                    loss.backward()
                    g_optimizer.step()
                self.update_all_score()
            # for d_epoch in trange(args.epochs_d):
            #     center_nodes, neighbor_nodes, labels = self.sample_for_d(args.n_sample_d, args.update_ratio)
            #     d_data += len(labels)
            #     for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, labels, args.bs_d):
            #         d_optimizer.zero_grad()
            #         loss = self.discriminator(*batch_data)
            #         d_loss.append(loss.item())
            #         loss.backward()
            #         d_optimizer.step()
            # for g_epoch in trange(args.epochs_g):
            #     center_nodes, neighbor_nodes = self.generate_window_pairs(
            #         self.sample_for_g(args.n_sample_g, args.update_ratio), args.gan_window_size)
            #     center_nodes = torch.LongTensor(center_nodes).to(self.device)
            #     neighbor_nodes = torch.LongTensor(neighbor_nodes).to(self.device)
            #     rewards = self.discriminator.get_reward(center_nodes, neighbor_nodes)
            #     g_data += len(rewards)
            #     for batch_data in self.get_batch_data(center_nodes, neighbor_nodes, rewards, args.bs_g, False):
            #         g_optimizer.zero_grad()
            #         loss = self.generator(*batch_data)
            #         g_loss.append(loss.item())
            #         loss.backward()
            #         g_optimizer.step()
            #     self.update_all_score()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_g, avr_loss_d = np.mean(g_loss), np.mean(d_loss)
                self.print(f'Epoch: {epoch:04d} d_loss: {avr_loss_d:.4f} d_data:{d_data} g_loss: {avr_loss_g:.4f} g_data:{g_data} duration: {duration:.2f}')
                self.stats['d_g_loss'].append((epoch, avr_loss_d))
                self.stats['g_g_loss'].append((epoch, avr_loss_g))
                d_loss, g_loss, d_data, g_data = [], [], 0, 0

            if epoch % args.save_every == 0:
                flag = self.evaluate(args, evaluate_funcs, epoch, d_optimizer, g_optimizer)
                if args.early_stop and flag:
                    break

        self.save_all(args)

    def find_category(self, node_embeddings, categorys=None):
        taxo_embed = self.init_taxo_embed(self.eval_model.get_embed())
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
        taxo_embed = self.init_taxo_embed(self.eval_model.get_embed())
        node_embed = self.eval_model.get_embed()
        for (node, category) in test_data:
            category_path = self.find_path(self.category2id[category])[:-1]
            level = len(category_path)
            embeds = node_embed[node].reshape(1, -1).repeat(level, axis=0)
            p = self.taxo_rootid
            for embed, true_pos in zip(embeds, category_path[::-1]):
                candidates = self.taxo_parent2children.get(p, None)
                if candidates is None:
                    continue
                pred = (embed @ taxo_embed[candidates].T).argmax()
                pred = candidates[pred]
                correct.append(pred == true_pos)
                if level_by_level:
                    p = true_pos
                else:
                    p = pred
        accs = np.array([np.sum(i) * 100 / len(i) for i in np.array_split(correct, split)])
        return accs.mean(), accs.std()