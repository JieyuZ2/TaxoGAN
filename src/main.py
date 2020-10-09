import sys
sys.path.append('./')
import os.path as osp
import argparse
from datetime import timedelta

from src.utils import *
from src.line import LINE
from src.pte import PTE
from src.deepwalk import DeepWalk
from src.graphsage import GraphSAGE
from src.graphgan import GraphGan
from src.taxogan import TaxoGAN_V1, TaxoGAN_V2, TaxoGAN_V3
from src.evaluation import evaluate_taxonomy, evaluate_node_classification, evaluate_link_prediction


def parse_args():
    parser = argparse.ArgumentParser()
    # general options
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['pubmed', 'dblp', 'yelp', 'freebase'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--prefix", type=str, default='', help="prefix use as addition directory")
    parser.add_argument('--suffix', default='', type=str, help='suffix append to log dir')
    parser.add_argument('--log_level', default=20)
    parser.add_argument('--log_every', type=int, default=1, help='log results every epoch.')
    parser.add_argument('--save_every', type=int, default=1, help='save learned embedding every epoch.')
    parser.add_argument("--embed_path", type=str, default='', help='path to save embedding.')

    # evaluation options
    parser.add_argument('--task', type=str, default='link_prediction', choices=['link_prediction', 'taxonomy'])
    parser.add_argument('--level_by_level', type=int, default=0)
    parser.add_argument('--eval_file', type=str, default='')

    # module options
    parser.add_argument('--model', type=str, default='GraphSAGE', help='[DeepWalk, LINE, GraphSAGE, GraphGan, PTE, TaxoGAN_V3, TaxoGAN_V1, TaxoGAN_V2].')
    parser.add_argument('--extend_label', type=int, default=1)
    parser.add_argument('--transform', type=int, default=1)
    parser.add_argument('--sibling_sample', type=int, default=1)
    # parser.add_argument('--top_down', type=int, default=0)
    # parser.add_argument('--stacked_transform', type=int, default=1)
    parser.add_argument('--rand_init', type=int, default=1, help="whether to randomly initialize embedding")
    parser.add_argument('--embed_dim', type=int, default=50)

    # training options
    parser.add_argument('--optimizer', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD'], default='Adam')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--early_stop', type=int, default=0)
    parser.add_argument('--minimal_epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=100)

    # for GAN
    parser.add_argument('--epochs_d', type=int, default=1)
    parser.add_argument('--epochs_g', type=int, default=1)
    parser.add_argument('--n_sample_d', type=int, default=20, help="the size of negative search space")
    parser.add_argument('--n_sample_g', type=int, default=4, help="the length of sampled trace for generator")
    parser.add_argument('--bs_d', type=int, default=64)
    parser.add_argument('--bs_g', type=int, default=64)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--update_ratio', type=float, default=1.0)
    parser.add_argument('--gan_window_size', type=int, default=1)
    parser.add_argument('--lambda_d', type=float, default=1e-4)
    parser.add_argument('--lambda_g', type=float, default=1e-4)
    parser.add_argument('--lambda_taxo', type=float, default=0.1)

    # for GCN
    parser.add_argument('--negative_power', type=float, default=0.75)
    parser.add_argument('--negative_sample_size', type=int, default=1)
    parser.add_argument('--neighbor_sample_size', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.2)

    # for DeepWalk
    parser.add_argument('--random_walk_path', type=str, default='', help='the path to save/load random walks.')
    parser.add_argument('--num_walker', type=int, default=10)
    parser.add_argument('--path_length', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0, help='restart rate.')
    parser.add_argument('--dw_window_size', type=int, default=2)

    return parser.parse_args()


def build_evaluate_func(args, task, eval_file=None):
    if task == 'node_classification':
        if not eval_file:
            eval_file = osp.join(args.data_dir, 'label.nf.dat')
        data = []
        labels = set()
        with open(eval_file) as fin:
            for l in fin.read().split('\n'):
                nid, label = l.split('\t')
                labels.add(label)
                data.append([int(nid), int(label)])
        args.num_class_eval = len(labels)
        data = np.array(data)
        np.random.shuffle(data)
        def evaluation_func(embed):
            return evaluate_node_classification(args, data, embed)

    elif task == 'link_prediction':
        if not eval_file:
            eval_file = osp.join(args.data_dir, 'label.lp.dat')
        args.link_file = osp.join(osp.dirname(eval_file), 'link.remain.lp.dat')
        if not (osp.isfile(eval_file) or osp.isfile(args.link_file)):
            eval_file, args.link_file = construct_eval_file_for_link_prediction(osp.join(args.data_dir, 'link.dat'))
        data = []
        with open(eval_file) as fin:
            for l in fin:
                n1, n2, label = l.strip().split('\t')
                data.append([int(n1), int(n2), int(label)])
        data = np.array(data)
        np.random.shuffle(data)
        def evaluation_func(embed):
            return evaluate_link_prediction(args, data, embed)

    elif task == 'taxonomy':
        if not eval_file:
            eval_file = osp.join(args.data_dir, 'label.taxo.dat')
        args.taxo_assign_file = osp.join(osp.dirname(eval_file), 'taxo.remain.taxo.dat')
        if not (osp.isfile(eval_file) or osp.isfile(args.taxo_assign_file)):
            eval_file, args.taxo_assign_file = construct_eval_file_for_taxo_evaluation(args.taxo_file, osp.join(args.data_dir, 'taxo.dat'), sample_ratio=0.1)
        with open(eval_file) as fin:
            data = []
            for l in fin:
                node, category = l.strip().split('\t')
                data.append((int(node), category))
        def evaluation_func(model):
            return evaluate_taxonomy(args, data, model, args.level_by_level)

    else:
        exit('Unsupported evaluation!')

    return eval_file, len(data), evaluation_func


def main(args):
    start_time = time.time()

    args.data_dir = osp.join('./data', args.dataset)
    args.link_file = osp.join(args.data_dir, 'link.dat')
    args.node_file = osp.join(args.data_dir, 'node.dat')
    args.taxo_assign_file = osp.join(args.data_dir, 'taxo.dat')
    args.taxo_file = osp.join(args.data_dir, 'taxo.json')

    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    evaluate_funcs = {}
    tasks = args.task.split('|')
    eval_files = args.eval_file.split('|')
    num_eval_datas = []
    if len(tasks)==len(eval_files):
        for task, eval_file in zip(tasks, eval_files):
            _, num_eval_data, evaluate_funcs[task] = build_evaluate_func(args, task, eval_file)
            num_eval_datas.append(num_eval_data)
    else:
        eval_files = []
        for task in tasks:
            eval_file, num_eval_data, evaluate_funcs[task] = build_evaluate_func(args, task)
            eval_files.append(eval_file)
            num_eval_datas.append(num_eval_data)
    args.eval_file = '|'.join(eval_files)
    args.num_eval_datas = num_eval_datas
    logger = init_logger(args)

    if not args.embed_path: args.embed_path = osp.join(args.log_dir, 'embed.txt')

    if args.model == 'TransformPTE' or args.sibling_sample:
        args.extend_label = 1

    print_config(args, logger)
    ModelClass = {'DeepWalk':DeepWalk, 'LINE':LINE, 'GraphSAGE':GraphSAGE, 'GraphGan':GraphGan, 'PTE':PTE, 'TaxoGAN_V1':TaxoGAN_V1, 'TaxoGAN_V2':TaxoGAN_V2, 'TaxoGAN_V3':TaxoGAN_V3}[args.model]
    model = ModelClass(args, logger)
    try:
        model.train(args, evaluate_funcs)
    except KeyboardInterrupt:
        model.save_all(args)

    logger.info("total cost time: {} ".format(timedelta(seconds=(time.time() - start_time))))


if __name__ == '__main__':
    args = parse_args()
    main(args)

