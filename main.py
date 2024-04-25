from __future__ import print_function

import argparse
import pdb
import os
import math
import sys
import pickle

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from datasets.dataset_custom import Custom_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def save_data_as_dict(subset, file_path):
    data_dict = {}
    for idx in subset.indices:
        name, features, _ = subset.dataset[idx]

        # print(subset.dataset[idx])
        if name not in data_dict:
            data_dict[name] = []
        
        data_dict[name].append({'feature': features.numpy()})
        # print(data_dict[name])
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)


def main(args):
    # create results directory if necessary
    if not os.path.isdir(os.path.join(args.results_dir,'_{}'.format(args.runcycle))):
        os.mkdir(os.path.join(args.results_dir,'_{}'.format(args.runcycle)))
    #
    # if args.k_start == -1:
    #     start = 0
    # else:
    #     start = args.k_star
    # t
    # if args.k_end == -1:
    #     end = args.k
    # else:
    #     end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []


    #k折交叉取消
    # for i in folds:
    # seed_torch(args.seed)
    # train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
    #         csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
    ###dataset需要重新规定
    if args.redivide == 0:
        # 1. 加载整个数据集。
        full_dataset = Custom_Dataset(os.path.join(args.data_root_dir, 'mDATA_train.pkl'))
        # 2. 根据数据集的大小，计算验证集的大小。
        val_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        # 3. 使用torch.utils.data.random_split将数据集分为训练集和验证集。
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # 分别保存训练集和验证集
        save_data_as_dict(train_subset, os.path.join(args.data_root_dir, 'trainset.pkl'))
        save_data_as_dict(val_subset, os.path.join(args.data_root_dir, 'valset.pkl'))
        _, features, _ = train_subset[0]
        print('before d train: {}'.format(features.numpy().shape))


        train_subset = Custom_Dataset(os.path.join(args.data_root_dir, 'trainset.pkl'))
        val_subset = Custom_Dataset(os.path.join(args.data_root_dir, 'valset.pkl'))
        _, features, _ = train_subset[0]
        print('after d train: {}'.format(features.numpy().shape))
    elif args.redivide == 1:
        train_subset = Custom_Dataset(os.path.join(args.data_root_dir, 'trainset.pkl'))
        val_subset = Custom_Dataset(os.path.join(args.data_root_dir, 'valset.pkl'))
    else:
        train_subset = Custom_Dataset(os.path.join(args.data_root_dir, 'mDATA_train.pkl'))
        val_subset = 0




    savedStdout = sys.stdout  #保存标准输出流
    print_log = open(os.path.join(args.results_dir,'_{}/'.format(args.runcycle), 'trainset.txt'),"w")
    sys.stdout = print_log
    print(train_subset[1])
    print(train_subset[2])
    # 中间print的内容都被输出到printlog.txt    
    sys.stdout = savedStdout  #恢复标准输出流
    print_log.close()


    test_dataset = Custom_Dataset(os.path.join(args.data_root_dir, 'mDATA_test.pkl'))
    _, features, _, _ = test_dataset[1]
    print('after d test: {}'.format(features.numpy().shape))

    datasets = (train_subset, val_subset, test_dataset)
    results, test_auc, val_auc, test_acc, val_acc, precision, recall, f1 = train(datasets, args)


    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)

    #write results to pkl
    # filename = os.path.join(args.results_dir, 'split_results.pkl')


    filename = os.path.join(args.results_dir,'_{}/'.format(args.runcycle), 'results.pkl')
    save_pkl(filename, results)

    np.set_printoptions(threshold=np.inf) #解决显示不完全问题

    f = open(filename,'rb')
    inf = pickle.load(f)
    f.close()
    inf = str(inf)
    ft = open(os.path.join(args.results_dir,'_{}/'.format(args.runcycle), 'resultstext.txt'),'w')
    ft.write(inf)
    ft.close()
    #
    # #
    #
    final_df = pd.DataFrame({'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc,
        'precision': precision, 'reacall': recall, 'f1': f1})
    print(final_df)

    savedStdout = sys.stdout  #保存标准输出流
    print_log = open(os.path.join(args.results_dir,'_{}/'.format(args.runcycle),'printlog.txt'),"w")
    sys.stdout = print_log
    print(final_df)
    # 中间print的内容都被输出到printlog.txt    

    sys.stdout = savedStdout  #恢复标准输出流
    print_log.close()


    # if len(folds) != args.k:
    #     save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    # else:
    #     save_name = 'summary.csv'
    # final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=1,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enable dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['abmil', 'clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--runcycle', type=int, default=1, help='numbr of running')
parser.add_argument('--redivide', type=int, default=2, help='choice of redevide val')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")






# def seed_torch(seed=7):
#     import random
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if device.type == 'cuda':
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#
# seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

# print('\nLoad Dataset')
#
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
#                             shuffle = False,
#                             seed = args.seed,
#                             print_info = True,
#                             label_dict = {'normal_tissue':0, 'tumor_tissue':1},
#                             patient_strat=False,
#                             ignore=[])
#
elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
#                             shuffle = False,
#                             seed = args.seed,
#                             print_info = True,
#                             label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
#                             patient_strat= False,
#                             ignore=[])
#
#     if args.model_type in ['clam_sb', 'clam_mb']:
#         assert args.subtyping
#
# else:
#     raise NotImplementedError
    
# if not os.path.isdir(args.results_dir):
#     os.mkdir(args.results_dir)
#
# args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
# if not os.path.isdir(args.results_dir):
#     os.mkdir(args.results_dir)
#
# if args.split_dir is None:
#     args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
# else:
#     args.split_dir = os.path.join('splits', args.split_dir)
#
# print('split_dir: ', args.split_dir)
# assert os.path.isdir(args.split_dir)
#
# settings.update({'split_dir': args.split_dir})


# with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
#     print(settings, file=f)
# f.close()
#
# print("################# Settings ###################")
# for key, val in settings.items():
#     print("{}:  {}".format(key, val))

if __name__ == "__main__":
    main(args)
    print("finished!")
    print("end script")


