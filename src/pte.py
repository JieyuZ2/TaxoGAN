import time
import functools
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.optim as optim
from collections import defaultdict

import src.utils as utils
from src.abstract_model import TaxoSkipGramModel, AbstractClass


class PTE(AbstractClass):
	def __init__(self, args, logger):
		super(PTE, self).__init__(args, logger)
		edge_dist_dict, node_dist_dict = utils.makeDist(args.link_file, args.negative_power)
		self.edges_alias_sampler = utils.VoseAlias(edge_dist_dict)
		self.nodes_alias_sampler = utils.VoseAlias(node_dist_dict)

		self.taxo_parent2children, self.taxo_child2parents, self.nodeid2category, self.category2nodeid, self.category2id, self.nodeid2path \
			= utils.read_taxos(args.taxo_file, args.taxo_assign_file, args.extend_label)

		edgedistdict, taxodistdict = self.make_dist()
		self.taxo_edge_alias_sampler = utils.VoseAlias(edgedistdict)
		self.taxo_alias_sampler = utils.VoseAlias(taxodistdict)
		self.num_category = len(self.category2id)
		self.taxo_rootid = self.category2id['root']

		self.category2level = utils.category2level(range(self.num_category), self.taxo_child2parents)
		level2category = defaultdict(list)
		for cate, level in self.category2level.items():
			level2category[level].append(cate)
		self.level2category = dict(level2category)

		self.print(f"num of category: {self.num_category}")

		if args.rand_init:
			taxo_init_embed = utils.init_embed(self.num_category, self.embed_dim)
		else:
			taxo_init_embed = self.init_taxo_embed(self.pre_train_embedding)

		self.model = TaxoSkipGramModel(self.pre_train_embedding, taxo_init_embed)
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

		batch_count = args.bs * (1 + args.negative_sample_size)
		batch_range = (self.num_node * 10) // batch_count
		data_num, taxo_data_num = 0, 0
		graph_loss, taxo_loss = [], []
		train_start_time = time.time()
		for epoch in range(args.epochs):
			for _ in trange(batch_range):
				data_num += batch_count
				optimizer.zero_grad()
				batch_data = self.sample_data(args.bs, args.negative_sample_size)
				loss = self.model(*batch_data)
				loss.backward()
				graph_loss.append(loss.item())
				optimizer.step()

			for _ in trange(batch_range):
				taxo_data_num += batch_count
				optimizer.zero_grad()
				batch_data = self.sample_taxo(args.bs, args.negative_sample_size)
				loss = args.lambda_taxo * self.model.forward_taxo(*batch_data)
				loss.backward()
				taxo_loss.append(loss.item())
				optimizer.step()

			if epoch % args.log_every == 0:
				duration = time.time() - train_start_time
				graph_avr_loss = np.mean(graph_loss)
				taxo_avr_loss = np.mean(taxo_loss)
				self.print(
					f'Epoch: {epoch:04d} graph loss: {graph_avr_loss:.4f} graph data:{data_num:d} taxo loss: {taxo_avr_loss:.4f} taxo data:{taxo_data_num:d} duration: {duration:.2f}')
				self.stats['graph_loss'].append((epoch, graph_avr_loss))
				self.stats['taxo_loss'].append((epoch, taxo_avr_loss))
				graph_loss, taxo_loss = [], []
				data_num, taxo_data_num = 0, 0

			if epoch % args.save_every == 0:
				flag = self.evaluate(args, evaluate_funcs, epoch, optimizer)
				if args.early_stop and flag:
					break

		self.save_all(args)

	def get_batch_data(self, left, right, label, batch_size, convert=True):
		if convert:
			for start in range(0, len(label), batch_size):
				end = start + batch_size
				yield torch.LongTensor(left[start:end]).to(self.device), torch.LongTensor(right[start:end]).to(
					self.device), torch.ByteTensor(label[start:end]).to(self.device)
		else:
			for start in range(0, len(label), batch_size):
				end = start + batch_size
				yield left[start:end], right[start:end], label[start:end]

	def sample_taxo(self, batch_size, negsample_size, sibling_sample=False):
		sampled_pairs = []
		labels = ([1] + [0] * negsample_size) * batch_size
		for (node, category) in self.taxo_edge_alias_sampler.sample_n(batch_size):
			sampled_pairs.append((node, category))
			negsample = 0
			while negsample < negsample_size:
				sampledcategory = self.taxo_alias_sampler.alias_generation()
				if sampledcategory in self.nodeid2category[node]:
				    continue
				else:
				    negsample += 1
				    sampled_pairs.append((node, sampledcategory))
				negsample += 1
				sampled_pairs.append((node, sampledcategory))
		return torch.LongTensor(sampled_pairs).to(self.device), torch.DoubleTensor(labels).to(self.device)

	def sample_data(self, batch_size, negsample_size):
		sampled_pairs = []
		labels = ([1] + [0] * negsample_size) * batch_size
		for (src_node, des_node) in self.edges_alias_sampler.sample_n(batch_size):
			if np.random.sample() > 0.5:
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
				negsample += 1
				sampled_pairs.append((src_node, samplednode))
		return torch.LongTensor(sampled_pairs).to(self.device), torch.DoubleTensor(labels).to(self.device)

	def find_category(self, node_embeddings, categorys=None):
		taxo_embed = self.model.get_taxo_embed()
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
		taxo_embed = self.model.get_taxo_embed()
		node_embed = self.model.get_embed()
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
		accs = np.array([np.sum(i) * 100 / len(i) for i in np.array_split(correct, split)])
		return accs.mean(), accs.std()
