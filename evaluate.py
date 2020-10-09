import os
import os.path as osp
import argparse

from src.utils import *
from src.evaluation import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument("--embed_path", type=str, default='log/results/dblp/LINE/embed.d.txt')
    parser.add_argument('--task', type=str, default='link_prediction', choices=['link_prediction', 'node_classification', 'taxo_assignment'])
    parser.add_argument('--eval_file', type=str, default='')
    return parser.parse_args()


def read_emd(filename):
    with open(filename, "r") as f:
        n_node, embed_dim = f.readline().strip().split()
        # for i, line in enumerate(f):
        #     emd = line.split()
        #     node_embed[i, :] = str_list_to_float(emd[1:])

        node_embed = []
        for i, line in enumerate(f):
            emd = line.split()
            node_embed.append(str_list_to_float(emd[1:]))
        node_embed = np.array(node_embed)
    return node_embed


def construct_eval_func(args):
    args.data_dir = osp.join('./data', args.dataset)

    if args.task == 'node_classification':
        if not args.eval_file:
            args.eval_file = osp.join(args.data_dir, 'label.nf.dat')
        data = []
        labels = set()
        with open(args.eval_file) as fin:
            for l in fin.read().split('\n'):
                nid, label = l.split('\t')
                labels.add(label)
                data.append([int(nid), int(label)])
        args.num_class_eval = len(labels)
        data = np.array(data)
        np.random.shuffle(data)

        def evaluation_func(embed):
            return evaluate_node_classification(args, data, embed)

    elif args.task == 'link_prediction':
        if not args.eval_file:
            args.eval_file = osp.join(args.data_dir, 'label.lp.dat')
        data = []
        with open(args.eval_file) as fin:
            for l in fin:
                n1, n2, label = l.strip().split('\t')
                data.append([int(n1), int(n2), int(label)])
        data = np.array(data)
        np.random.shuffle(data)

        def evaluation_func(embed):
            return evaluate_link_prediction(args, data, embed, logistic_regression=True)

    elif args.task == 'taxo_assignment':
        if not args.eval_file:
            args.eval_file = osp.join(args.data_dir, 'label.ta.dat')
        with open(args.eval_file) as fin:
            label2id = {}
            data = []
            cnt = 0
            next(fin)
            for l in fin:
                if l == 'test\n':
                    train_data, data = np.array(data), []
                    continue
                n, label = l.strip().split('\t')
                if label not in label2id:
                    label2id[label] = cnt
                    cnt += 1
                data.append([int(n), label2id[label]])
            test_data = np.array(data)
        args.num_class_eval = len(label2id)

        def evaluation_func(embed):
            return evaluate_taxo_assignment(args, train_data, test_data, embed)
    else:
        exit('Unsupported evaluation!')

    return evaluation_func


if __name__ == '__main__':
    args = parse_args()
    eval_func = construct_eval_func(args)

    if osp.isdir(args.embed_path):
        for file in os.listdir(args.embed_path):
            file = osp.join(args.embed_path, file)
            try:
                embed = read_emd(file)
            except Exception as e:
                print(f'{file} may not be embedding file')
                continue
            acc, std = eval_func(embed)
            print(f'{file}: acc={acc:.2f}+-{std:.2f}')
    elif osp.isfile(args.embed_path):
        embed = read_emd(args.embed_path)
        acc, std = eval_func(embed)
        print(f'{args.embed_path}: acc={acc:.2f}+-{std:.2f}')
