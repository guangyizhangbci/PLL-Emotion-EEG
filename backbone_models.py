#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:01:00 2022

@author: patrick
"""
from __future__ import print_function
import numpy as np
import math
from math import log, e
from scipy.stats import *
import copy
import PIL
from IPython.display import HTML
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class conv_EEG(nn.Module):
    def __init__(self):
        super(conv_EEG, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, 3, stride=1)
        self.bn1   = nn.BatchNorm1d(5)
        self.lr1   = nn.LeakyReLU(0.3)
        self.mxp   = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(5, 10, 3, stride=1)
        self.bn2   = nn.BatchNorm1d(10)
        self.lr2   = nn.LeakyReLU(0.3)
        self.fn    = nn.Flatten()


        self.classifier = nn.Sequential(

                            nn.Linear(3060,64),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(64, 5))


    def forward(self, x):
        output = self.conv1(x)

        output = self.bn1(output)
        output = self.lr1(output)

        # output = self.mxp(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.lr2(output)

        # output = self.mxp(output)
        # decoded_output = self.decoder(encoded_output)
        output = self.fn(output)

        output = self.classifier(output)


        return output




class conv_EEG_pico(nn.Module):
    def __init__(self):
        super(conv_EEG_pico, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, 3, stride=1)
        self.bn1   = nn.BatchNorm1d(5)
        self.lr1   = nn.LeakyReLU(0.3)
        self.mxp   = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(5, 10, 3, stride=1)
        self.bn2   = nn.BatchNorm1d(10)
        self.lr2   = nn.LeakyReLU(0.3)
        self.fn    = nn.Flatten()
        self.fc    = nn.Linear(64, 5)


        self.features = nn.Sequential(
                            nn.Linear(3060,64),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            )


    def forward(self, x):
        output = self.conv1(x)

        output = self.bn1(output)
        output = self.lr1(output)

        # output = self.mxp(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.lr2(output)

        # output = self.mxp(output)
        # decoded_output = self.decoder(encoded_output)
        output = self.fn(output)

        features = self.features(output)
        logits   = self.fc(features)

        return logits, F.normalize(features, dim=1)



class PiCO(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()

        self.encoder_q = base_encoder()
        # momentum encoder
        self.encoder_k = base_encoder()

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue, args.num_class))
        self.register_buffer("queue_partial", torch.randn(args.moco_queue, args.num_class))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, partial_Y, args):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size, :]  = labels
        self.queue_partial[ptr:ptr + batch_size, :] = partial_Y
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr


    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes

    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):

        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # for testing

        predicted_scores = torch.softmax(output, dim=1) * partial_Y
        max_scores, pseudo_labels = torch.max(predicted_scores, dim=1)
        # using partial labels to filter out negative labels

        # compute protoypical logits
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        # update momentum prototypes with pseudo labels
        for feat, label, max_score in zip(q, pseudo_labels, max_scores):
            self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
        # normalize prototypes
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            # im_k, predicted_scores, partial_Y, idx_unshuffle = self._batch_shuffle_ddp(im_k, predicted_scores, partial_Y)
            _, k = self.encoder_k(im_k)
            # undo shuffle
            # k, predicted_scores, partial_Y = self._batch_unshuffle_ddp(k, predicted_scores, partial_Y, idx_unshuffle)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_scores  = torch.cat((predicted_scores, predicted_scores, self.queue_pseudo.clone().detach()), dim=0)
        partial_target = torch.cat((partial_Y, partial_Y, self.queue_partial.clone().detach()), dim=0)
        # to calculate SupCon Loss using pseudo_labels and partial target

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, predicted_scores, partial_Y, args)

        return output, features, pseudo_scores, partial_target, score_prot
















#
