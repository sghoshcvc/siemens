'''
Created on Jan 20, 2020

@author: Suman Ghosh

- works on vehicle dataset
- load dataset
- 80%, 10%m 10%
- save and load hardcoded name 'cars.pt'
'''
import argparse
import logging
import sys


import torch.autograd
import torch.cuda
import torch.optim

from torch.utils.data import DataLoader

sys.path.insert(0, '/home/suman/siemens/')
import copy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from dataset.ctrda import CtRdaDataset
from model.model import initialize_model
from model.spp import SPPNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from tools.test import evaluate_cnn_batch


def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def train():
    logger = logging.getLogger('Siemens-Experiment::train')
    logger.info('--- Running Siemens Training ---')

    # argument parsing
    parser = argparse.ArgumentParser()
    # - train arguments
    parser.add_argument('--learning_rate_step', '-lrs', type=learning_rate_step_parser, default='30000:1e-5,60000:1e-5,150000:1e-5',
                        help='A dictionary-like string indicating the learning rate for up to the number of iterations. ' +
                             'E.g. the default \'70000:1e-4,80000:1e-5\' means learning rate 1e-4 up to step 70000 and 1e-5 till 80000.')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the Simens. Default: 500')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=16,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--test_batch_size', '-tbs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    #parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default=0,
    #                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='0',
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    # - experiment arguments

    args = parser.parse_args()

    # sanity checks
    if not torch.cuda.is_available():
        logger.info('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # prepare datset loader
    #TODO: add augmentation
    # logger.info('Loading dataset %s...', args.dataset)
    # if args.dataset == 'gw':
    #     train_set = GWDataset(gw_root_dir='../../../pytorch-phocnet/data/gw',
    #                           cv_split_method='almazan',
    #                           cv_split_idx=1,
    #                           image_extension='.tif',
    #                           embedding=args.embedding_type,
    #                           phoc_unigram_levels=args.phoc_unigram_levels,
    #                           fixed_image_size=args.fixed_image_size,
    #                           min_image_width_height=args.min_image_width_height)


    train_set = CtRdaDataset(data_dir= '../data/Siemens_CT_RDA',
                               image_extension='.jpg')
    test_set = copy.copy(train_set)
    val_set = copy.copy(train_set)

    train_set.mainLoader(partition='train')
    test_set.mainLoader(partition='test')
    val_set.mainLoader(partition='val')

    train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=8)

    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=8)
    val_loader = DataLoader(val_set,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=8)
    test_interval =500

    # model_ft, input_size = initialize_model(train_set.nclasses, feature_extract=True, use_pretrained=True)

    model_ft = SPPNet(n_out=train_set.nclasses,
                 input_channels=3,
                 gpp_type='spp')

    criterion = nn.CrossEntropyLoss(size_average=False)

    print('model created')

    model_ft.cuda()
    early_stopping_cnt = 0
    early_stopping_flag = False
    best_acc = 0
    optim = torch.optim.Adam(model_ft.parameters(), lr=0.001)

    for epoch in range(256):

        for batch_idx, data_batch in enumerate(train_loader):
            optim.zero_grad()
            images, class_ids = data_batch
            images = Variable(images.float().cuda())
            labels = Variable(class_ids.cuda())
            pred_classes = model_ft(images)
            loss = criterion(pred_classes,labels)
            loss.backward()
            optim.step()

            if batch_idx % test_interval:
                logger.info('Evaluating net after %d iterations', epoch)
                accuracy,sensitivity,specificity = evaluate_cnn_batch(cnn=model_ft, dataset_loader=val_loader)
                logger.info(' Validation Accuracy after %d iterations: %3.2f', batch_idx, accuracy[0])
                logger.info('Evaluating net after %d iterations', epoch)
                accuracy,sensitivity,specificity = evaluate_cnn_batch(cnn=model_ft, dataset_loader=test_loader)
                logger.info('Test Accuracy after %d iterations: %3.2f', batch_idx, accuracy[0])


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()