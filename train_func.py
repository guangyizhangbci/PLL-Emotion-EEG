"""
Created on Sat Oct 22 19:10:11 2022

@author: patrick
"""

import torch
import numpy as np
import torch.nn.functional as F
import parsing
from utils import *

parser = parsing.create_parser()
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


'''
***On the Power of Deep but Naive Partial Label Learning (DNPL), ICASSP 21***
'''
class T_DNPL():
    def __init__(self):
        super(T_DNPL, self).__init__()

    def train_step(self, index, confidence, input, input_w, input_s, part_y, model, optimizer, epoch):

        optimizer.zero_grad()

        output  = model(input)
        s = part_y
        s_hat = F.softmax(output, dim=1)
        ss_hat = s * s_hat
        ss_hat_dp = ss_hat.sum(1)
        ss_hat_dp = torch.clamp(ss_hat_dp, 0., 1.)
        loss = -torch.mean(torch.log(ss_hat_dp + 1e-10))

        loss.backward()

        optimizer.step()


        return loss.item(), confidence


'''
***Progressive Identification of True Labels for Partial-Label Learning (PRODEN), ICML 20***
'''

class T_PRODEN():
    def __init__(self):
        super(T_PRODEN, self).__init__()

    def confidence_update(self, index, confidence, outputs, part_y):

        revisedY = confidence[index, :].clone()
        revisedY[revisedY > 0]  = 1
        revisedY = revisedY * outputs
        revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

        confidence[index, :] = revisedY.detach()

        return confidence


    def train_step(self, index, confidence, input, input_w, input_s, part_y, model, optimizer, epoch):

        optimizer.zero_grad()

        outputs      =  F.softmax(model(input), dim=1)
        l = confidence[index, :] * torch.log(outputs)

        loss  = (-torch.sum(l)) / l.size(0)

        loss.backward()
        optimizer.step()

        if args.use_confidence==True:
            confidence = self.confidence_update(index, confidence, outputs, part_y)

        return loss.item(), confidence


'''
***Exploiting Class Activation Value for Partial-Label Learning (CAVL), ICLR 22***
'''

class T_CAVL():
    def __init__(self):
        super(T_CAVL, self).__init__()

    def confidence_update(self, index, confidence, outputs, part_y):

        with torch.no_grad():
            cav = (outputs*torch.abs(1-outputs))*part_y
            cav_pred = torch.max(cav,dim=1)[1]
            gt_label = F.one_hot(cav_pred, part_y.shape[1]) # label_smoothing() could be used to further improve the performance for some datasets
            confidence[index, :] = gt_label.float()

        return confidence

    def train_step(self, index, confidence, input, input_w, input_s, part_y, model, optimizer, epoch):

        optimizer.zero_grad()

        outputs       = model(input)
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * confidence[index, :]

        loss = - ((final_outputs).sum(dim=1)).mean()

        loss.backward()
        optimizer.step()

        if args.use_confidence==True:
            confidence = self.confidence_update(index, confidence, outputs, part_y)

        return loss.item(), confidence


'''
***Leveraged Weighted Loss for Partial Label Learning (LW), ICML 21***
'''

class T_LW():
    def __init__(self):
        super(T_LW, self).__init__()


    def confidence_update(self, index, confidence, outputs, part_y):

        with torch.no_grad():

            sm_outputs = F.softmax(outputs, dim=1)

            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[part_y > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(device)
            counter_onezero = counter_onezero.to(device)

            new_weight1 = sm_outputs * onezero
            new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight2 = sm_outputs * counter_onezero
            new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
                confidence.shape[1], 1).transpose(0, 1)
            new_weight  = new_weight1 + new_weight2

            confidence[index, :] = new_weight

            return confidence


    def train_step(self, index, confidence, input, input_w, input_s, part_y, model, optimizer, epoch):

        optimizer.zero_grad()

        outputs = model(input)

        device = outputs.device

        onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
        onezero[part_y > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        # loss 1 is applied on candidate labels and loss 2 is applied on non-candidate labels.
        if args.loss == 'sigmoid':

            sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
            sig_loss1 = sig_loss1.to(device)
            sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))

            sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
                1 + torch.exp(-outputs[outputs > 0]))
            if args.use_confidence==True:
                l1 = confidence[index, :] * onezero * sig_loss1
            else:
                l1 = onezero * sig_loss1

            average_loss1 = torch.sum(l1) / l1.size(0)

            sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
            sig_loss2 = sig_loss2.to(device)
            sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
            sig_loss2[outputs < 0] = torch.exp(
                outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
            if args.use_confidence==True:
                l2 = confidence[index, :] * counter_onezero * sig_loss2
            else:
                l2 = counter_onezero * sig_loss2

            average_loss2 = torch.sum(l2) / l2.size(0)

            loss = average_loss1 + args.beta * average_loss2


        elif args.loss == 'cross_entropy':

            sm_outputs = F.softmax(outputs, dim=1)

            sig_loss1 = - torch.log(sm_outputs + 1e-8)
            if args.use_confidence==True:
                l1 = confidence[index, :] * onezero * sig_loss1
            else:
                l1 = onezero * sig_loss1
            average_loss1 = torch.sum(l1) / l1.size(0)

            sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
            if args.use_confidence==True:
                l2 = confidence[index, :] * counter_onezero * sig_loss2
            else:
                l2 = counter_onezero * sig_loss2
            average_loss2 = torch.sum(l2) / l2.size(0)

            loss = average_loss1 + args.beta * average_loss2

        else:
            raise Exception('Need to choose the loss')

        loss.backward()
        optimizer.step()



        if args.use_confidence==True:
            confidence = self.confidence_update(index, confidence, outputs, part_y)

        return loss.item(), confidence



'''
***Revisiting Consistency Regularization for Deep Partial Label Learning (CR), ICML 22***
'''

class T_CR():
    def __init__(self):
        super(T_CR, self).__init__()


    def confidence_update(self, index, confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y):

        part_y[part_y>0]=1
        y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas = map(lambda x: x.detach(), (y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas))


        revisedY0 = part_y.clone()

        revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                    * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                    * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
        revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(args.num_class, 1).transpose(0, 1)

        confidence[index, :] = revisedY0.detach()

        return confidence



    def train_step(self, index, confidence, input, input_w, input_s, part_y, model, optimizer, epoch):
        
        part_y[part_y>0]=1
        consistency_criterion = nn.KLDivLoss(reduction='batchmean').to(device)

        optimizer.zero_grad()

        output, weak_output, strong_output  = map(lambda x: model(x), (input, input_w, input_s))

        consistency_loss, consistency_loss_weak, consistency_loss_strong =\
        map(lambda x: consistency_criterion(torch.log_softmax(x, dim=-1), confidence[index,:]), (output, weak_output, strong_output))


        super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(output, dim=1)) * (1 - part_y), dim=1))


        if args.use_confidence==True:
            lam = min((epoch / args.epochs) * args.lam, args.lam)
        else:
            lam = 0

        loss = super_loss + lam*(args.c_weight * consistency_loss + args.c_weight_w * consistency_loss_weak + args.c_weight_s * consistency_loss_strong)
  

        loss.backward()
        optimizer.step()
        if args.use_confidence==True:
            confidence = self.confidence_update(index, confidence, torch.softmax(output, dim=-1), torch.softmax(weak_output, dim=-1),
                             torch.softmax(strong_output, dim=-1), part_y)


        return loss.item(), confidence


'''
***PiCO: Contrastive Label Disambiguation for Partial Label Learning (PiCO), ICLR 22***
'''

class T_PiCO():
    def __init__(self):
        super(T_PiCO, self).__init__()


    def train_step(self, index, confidence, input, input_w, input_s, part_y, model, optimizer, epoch):

        loss_fn = partial_loss(confidence)
        loss_cont_fn = SupConLoss()

        cls_out, features_cont, pseudo_score_cont, partial_target_cont, score_prot = model(input, input, part_y, args, eval_only=False)

        pseudo_target_max, pseudo_target_cont = torch.max(pseudo_score_cont, dim=1)
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        if args.use_confidence==True:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=part_y)
            # warm up ended
            mask = torch.eq(pseudo_target_cont[:args.batch_size], pseudo_target_cont.T).float().cuda()
            # get positive set by contrasting predicted labels
        else:
            mask = None

        # contrastive loss
        loss_cont = loss_cont_fn(features=features_cont, mask=mask, batch_size=args.batch_size)
        # classification loss
        loss_cls = loss_fn(cls_out, index)


        loss = loss_cls + args.gamma * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#
        return loss.item(), confidence















#
