"""
Created on Sat May 14 15:01:14 2022

@author: patrick
"""
from __future__ import print_function
import numpy as np
import math
import random
from math import log, e
from tqdm import tqdm
from scipy.stats import *
import copy
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os,sys,inspect
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.preprocessing import MinMaxScaler
import argparse
from train_func import *
from utils import *
from backbone_models import conv_EEG, conv_EEG_pico, PiCO

data_addr  = './DATA/SEED_V/EEG/de_{}_{}.npy'      # subject_No, Fold_No
label_addr = './DATA/SEED_V/EEG/label_{}_{}.npy'   # subject_No, Fold_No


import parsing

parser = parsing.create_parser()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]

train_func_dict = {'DNPL': T_DNPL, 'PRODEN': T_PRODEN, 'CAVL': T_CAVL, 'LW': T_LW, 'CR': T_CR, 'PiCO': T_PiCO}


def eval_step(inputs, labels, model):

    if args.method =='PiCO':
        outputs_classification = model(inputs, args, eval_only=True)
    else:
        outputs_classification = model(inputs)

    classification_pred = torch.max(outputs_classification, 1)[1]


    batch_size = labels.shape[0]
    digital_labels = torch.max(labels, 1)[1]
    # print(digital_labels.shape)

    err = nn.CrossEntropyLoss()(outputs_classification, digital_labels)

    running_corrects = (classification_pred == digital_labels).float().sum()
    accuracy = running_corrects/batch_size
    # err = nn.CrossEntropyLoss()(classification_output, digital_labels)
    return err.item(), accuracy.detach().cpu().clone().numpy()



def train(Net, train_dataset, test_dataset):


    train_loss_epoch  = np.ones((args.epochs,  1))
    test_loss_epoch   = np.ones((args.epochs,  1))
    test_acc_epoch    = np.zeros((args.epochs, 1))

    'Label confidence initialization'

    confidence = copy.deepcopy(train_dataset.dataset.partial_labels)
    confidence = confidence / confidence.sum(axis=1)[:, None]

    confidence = torch.FloatTensor(confidence).to(device)

    'Choice of optmizer'
    if args.optimizer == 'adam':
        optimizer = optim.Adam(Net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(Net.parameters(), momentum=0.9, lr=args.lr, weight_decay=1e-4)
    else:
        raise Exception('Need to choose the optimizer')

    'Choice of using learning scheduler'
    if args.use_scheduler==True:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20], last_epoch=-1)



    '''
    ***Start of Training***
    '''

    for epoch in range(args.epochs):
        start = time.time()
        train_loss_batch = []
        train_acc_batch = []


        Net.train()


        for index, input, input_w, input_s, _, part_y in train_dataset:

            index, input, input_w, input_s, part_y= map(lambda x: x.to(device), (index, input, input_w, input_s, part_y))

            loss, confidence = train_func_dict[args.method]().train_step(index, confidence, input, input_w, input_s, part_y, Net, optimizer, epoch)

            train_loss_batch.append(loss)


            if args.use_scheduler==True:
                scheduler.step()

            if args.method =='PiCO':
                loss_fn = partial_loss(confidence)
                loss_fn.set_conf_ema_m(epoch, args)

        train_loss_epoch[epoch] = Average(train_loss_batch)


        Net.eval()

        with torch.no_grad():
            test_loss_batch = []
            test_acc_batch  = []

            for _, image_batch, label_batch, _ in test_dataset:
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                loss, acc = eval_step(image_batch,label_batch, Net)

                test_loss_batch.append(loss)
                test_acc_batch.append(acc)


        test_loss_epoch[epoch] = Average(test_loss_batch)
        test_acc_epoch[epoch]  = Average(test_acc_batch)


    best_test_acc = test_acc_epoch[np.argmin(train_loss_epoch)]


    return best_test_acc



if __name__ == '__main__':


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # results save path, depends on method and the arguments in parsing

    main_path = './SEED_V_result/PLL/main/' + args.method + '/scheduler_{}'.format(args.use_scheduler) + '/optimizer_{}/lr_{}/'.format(args.optimizer, args.lr)

    directory = main_path

    if args.method == 'DNPL':
        directory = main_path
    elif args.method == 'PRODEN' or args.method == 'CAVL':
        directory = main_path + 'confidence_{}/'.format(args.use_confidence)
    elif args.method == 'LW':
        directory = main_path + 'confidence_{}/'.format(args.use_confidence) + '{}/'.format(args.loss) + 'beta_{}/'.format(args.beta)
    elif args.method == 'CR':
        directory = main_path + 'confidence_{}/'.format(args.use_confidence) + 'weight_{}_{}_{}/'.format(c_weight, c_weight_w, c_weight_s)
    elif args.method == 'PiCO':
        directory = main_path + 'confidence_{}/'.format(args.use_confidence) + 'contrast_weight_{}/'.format(args.gamma)
    else:
        raise Exception('Need to choose the method')


    directory = directory + 'run_{}/'.format(args.run_idx)


    '''
    EEG Experiment Setup, do not change this part ------START
    '''

    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass

    '''Repeat Experiment Five Times'''

    random_seed_arr = [100, 42, 19, 57, 598]
    seed = random_seed_arr[args.run_idx-1]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.deterministic = True
    cudnn.benchmark = True


    if args.partial_type=='uniform':
        prob_arr =  [0.20, 0.40, 0.60, 0.80, 0.9, 0.95]
    elif args.partial_type=='emotion':
        prob_arr = [0]
    else:
        raise Exception('Need to choose the parital label generation method')


    acc_array = np.zeros((16, 3))

    for prob in prob_arr:
        if args.partial_type=='uniform':
            if os.path.exists(os.path.join(directory, "prob_{}.csv".format(prob))):
                continue
        elif args.partial_type=='emotion':
            if os.path.exists(os.path.join(directory, "gamma.csv")):
                continue
        else:
            raise Exception('Need to choose the parital label generation method')

        for subject_num in range(1, 17):

            # data and labels load

            X1 = np.load(data_addr.format(subject_num, 1))
            X2 = np.load(data_addr.format(subject_num, 2))
            X3 = np.load(data_addr.format(subject_num, 3))

            X  = np.vstack((X1, X2, X3))

            Y1 = np.load(label_addr.format(subject_num, 1))
            Y2 = np.load(label_addr.format(subject_num, 2))
            Y3 = np.load(label_addr.format(subject_num, 3))

            Y  = np.vstack((Y1, Y2, Y3))

            # data normalization

            scaler=MinMaxScaler()
            X = scaler.fit_transform(X)


            for fold_num in range(3):

                Net = conv_EEG().to(device)
                if args.method == 'PiCO':
                    Net = PiCO(args, conv_EEG_pico).to(device)

                #
                Net.apply(WeightInit)
                Net.apply(WeightClipper)


                fold_1_index = [i for i in range(0, len(X1))]
                fold_2_index = [i for i in range(len(X1), len(X1)+len(X2))]
                fold_3_index = [i for i in range(len(X1)+len(X2), len(X1)+len(X2)+len(X3))]
                
                # three-fold cross-validaiton based on pre-defined folds
                if fold_num ==0:
                    train_index, test_index = fold_1_index + fold_2_index, fold_3_index
                elif fold_num ==1:
                    train_index, test_index = fold_2_index + fold_3_index, fold_1_index
                else:
                    train_index, test_index = fold_3_index + fold_1_index, fold_2_index


                X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
                Y_train, Y_test = map(lambda x: to_categorical(np.ravel(x)), (Y_train, Y_test))

  
                if args.partial_type=='uniform':
                    partial_label_train, avgC = partialize(Y_train, p=prob) # generation of candidate labels obeys uniform distribution, value of p can be 0.2,0.4,0.6,0.8,0.9,0.95 

                elif args.partial_type=='emotion':
                    partial_label_train, avgC = partialize_emotion(Y_train, Y[train_index]) # generation of candidate labels depends on emotion similarities
                else:
                    break
                partial_label_test,  avgC = partialize(Y_test,  p=0.0) # value of p can be 0.4,0.6,0.8

                data_train, data_test  = np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1)
                label_train, label_test = Y_train, Y_test

                # data loader
                train_dataset  = load_augmented_dataset_to_device(data_train, label_train, partial_label_train, batch_size=8, shuffle_flag=True,  augmentation_flag=True)
                test_dataset   = load_augmented_dataset_to_device(data_test,  label_test,  partial_label_test,  batch_size=8, shuffle_flag=False, augmentation_flag=False)

                acc_array[subject_num-1, fold_num] = train(Net, train_dataset, test_dataset)


                torch.cuda.empty_cache()

        # save results
        if args.partial_type=='uniform':
            np.savetxt(os.path.join(directory, "prob_{}.csv".format(prob)),     acc_array , delimiter=",")
        elif args.partial_type=='emotion':
            np.savetxt(os.path.join(directory, "gamma.csv"),                    acc_array , delimiter=",")
        else:
            raise Exception('Need to choose the parital label generation method')
































    #
