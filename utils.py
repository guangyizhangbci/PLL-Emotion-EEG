"""
Created on Sat May 14 13:28:07 2022

@author: patrick
"""
import numpy as np
import copy
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import math


def Average(lst):
    return sum(lst) / len(lst)

def to_categorical(y):
    """ 1-hot encodes a tensor """
    num_classes = len(np.unique(y))
    return np.eye(num_classes, dtype='uint8')[y.astype(int)]


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

class WeightInit(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        torch.manual_seed(0)
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = nn.init.normal_(W, 0.0, 0.02)



def emotion_similarity():  # emotion similarity scores generation

    fear = [1, 117]
    disgust = [1, 153]
    sad = [1, 198]
    neutral = [0, 0]
    happy = [1, 18]


    matrix = np.eye(5)

    list = [fear, disgust, sad, neutral, happy]

    for i in range(5):
        for j in range(5):
            matrix[i,j] = 1 - math.sqrt(list[i][0]**2 + list[j][0]**2 -2*list[i][0]*list[j][0]*math.cos(math.radians(list[i][1])-math.radians(list[j][1]))) / 2

    return matrix



def partialize(y, p): # generation of candidate labels based on the uniform distribution
    new_y = copy.deepcopy(y).astype(float)
    n, c = y.shape[0], y.shape[1]
    avgC = 0

    for i in range(n):
        row = new_y[i, :]
        row[np.where(np.random.binomial(1, p, c)==1)] = 1
        while np.sum(row) == 1:
            row[np.random.randint(0, c)] = 1
        avgC += np.sum(row)
        new_y[i] = row / np.sum(row)

    avgC = avgC / n
    return new_y, avgC



def partialize_pair(y, y0, p):
    new_y = copy.deepcopy(y).astype(float)
    n, c = y.shape[0], y.shape[1]

    avgC = 0

    P = np.eye(c)
    for idx in range(0, c-1):
        P[idx, idx], P[idx, idx+1] = 1, p
    P[c-1, c-1], P[c-1, 0] = 1, p
    for i in range(n):
        row = new_y[i, :]
        idx = int(y0[i])
        row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1
        avgC += np.sum(row)

        new_y[i] = row / np.sum(row)

    avgC = avgC / n
    return new_y, avgC




def partialize_emotion(y, y0): # generation of candidate labels based on emotion similarities
    new_y = copy.deepcopy(y).astype(float)
    n, c = y.shape[0], y.shape[1]


    avgC = 0
    matrix = emotion_similarity()

    for i in range(n):
        row = new_y[i, :]

        for j in range(5):
            row[j] = np.random.binomial(1,  matrix[int(y0[i]), j], 1)


        while np.sum(row) == 1:
            row[np.random.randint(0, c)] = 1
        avgC += np.sum(row)

        new_y[i] = row / np.sum(row)



    avgC = avgC / n
    return new_y, avgC



def add_gaussian_noise_torch(input, std):

    input_shape =input.size()
    noise = torch.normal(mean=0.5, std=std, size =input_shape)
    # noise = noise.to(device)

    return input + noise

class CustomEEGDataset(Dataset): #customize dataset, containing data augmention which is required in methods CR and PiCO
    def __init__(self, image, labels, partial_labels, augmentation=True):
        self.data = image
        self.labels  = labels
        self.partial_labels = partial_labels
        self.augmentation  = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label, partial_label = map(lambda x:  torch.Tensor(x[index]), (self.data, self.labels, self.partial_labels))


        if self.augmentation is True:
            weak_aug   = add_gaussian_noise_torch(data, 0.2)
            strong_aug = add_gaussian_noise_torch(data, 0.8)

            output = index, data, weak_aug, strong_aug, label, partial_label
        else:
            output = index, data, label, partial_label

        return output


def load_dataset_to_device(data, label, partial_label, batch_size, shuffle_flag=True):


    data, label, partial_label = torch.Tensor(data), torch.Tensor(label), torch.Tensor(partial_label)

    dataset = torch.utils.data.TensorDataset(data, label, partial_label)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=shuffle_flag, num_workers=2, drop_last=True,  pin_memory=True)

    return dataset




def load_augmented_dataset_to_device(data, label, partial_label, batch_size, shuffle_flag=True, augmentation_flag=True):



    dataset = CustomEEGDataset(data, label, partial_label, augmentation_flag)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=shuffle_flag, num_workers=2, drop_last=True,  pin_memory=True)

    return dataset



class partial_loss(nn.Module): 
    """The supervised loss of PiCO with and without prototype-based label disambiguation"""
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        # logsm_outputs = F.softmax(outputs, dim=1)

        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - self.conf_ema_m) * pseudo_label
        return None




class SupConLoss(nn.Module): # Contrastive Loss used in PiCO
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )

            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n',    [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk',   [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#
