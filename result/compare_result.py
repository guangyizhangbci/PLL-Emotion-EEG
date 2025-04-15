import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

directory = '/home/patrick/Desktop/SEED_V_result/PLL/main'



#
# arr = [1, 3, 5,7, 10, 25]
# def get_result(path):
#     results = np.zeros((len(arr), 5))
#     for i in range(len(arr)):
#         for j in range(1, 6):
#             results[i,j-1] = np.average(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
#     return np.mean(results,1), np.std(results,1)
#

fig= plt.figure()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 8})




optimizer_arr = ['sgd']

lr_arr = [0.1, 0.01, 0.001]
wt_arr = ['warm_up', 'weight/0.5', 'weight/1.0', 'weight/2.0', 'weight/5.0']

prob_arr = [0.2, 0.4, 0.6, 0.8]


color_dict = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:green',
              'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
              'tab:cyan']

#
#
# color_idx = 0
# counter=0
# for optimizer in optimizer_arr:
#     for lr in lr_arr:
#         result_ICASSP=[]
#         result_ICASSP_s=[]
#
#         result_ICML=[]
#         result_ICLR=[]
#         ax = fig.add_subplot(1, 3, counter+1)
#         for wt in wt_arr:
#             result_ICLR_mixup=[]
#
#             for prob in prob_arr:
#
#                 # result_path_ICASSP21    = np.loadtxt(os.path.join(directory, 'ICASSP21/optimizer_{}/lr_{}/prob_{}.csv'.format(optimizer, lr, prob)), delimiter=",")
#                 # result_path_ICASSP21_s  = np.loadtxt(os.path.join(directory, 'ICASSP21/scheduler/optimizer_{}/lr_{}/prob_{}.csv'.format(optimizer, lr, prob)), delimiter=",")
#                 #
#                 # result_path_ICML20      = np.loadtxt(os.path.join(directory, 'ICML20/optimizer_{}/lr_{}/prob_{}.csv'.format(optimizer, lr, prob)), delimiter=",")
#
#                 # result_path_ICLR22      = np.loadtxt(os.path.join(directory, 'ICLR22/optimizer_{}/lr_{}/prob_{}.csv'.format(optimizer, lr, prob)), delimiter=",")
#                 result_path_ICLR22_mixup= np.loadtxt(os.path.join(directory, 'ICLR22_mixup/scheduler_True/optimizer_{}/lr_{}/{}/prob_{}.csv'.format(optimizer, lr, wt, prob)), delimiter=",")
#
#                 #
#                 # result_ICASSP.append(np.average(result_path_ICASSP21))
#                 # result_ICASSP_s.append(np.average(result_path_ICASSP21_s))
#                 #
#                 # result_ICML.append(np.average(result_path_ICML20))
#
#                 # result_ICLR.append(np.average(result_path_ICLR22))
#                 result_ICLR_mixup.append(np.average(result_path_ICLR22_mixup))
#
#
#             ax.title.set_text('{} lr{}'.format(optimizer, lr))
#
#
#             # ax.plot(prob_arr, result_ICASSP,     marker = '*',     label = 'w/o Scheduler ' + ' {} lr_{}'.format(optimizer, lr))
#             # ax.plot(prob_arr, result_ICASSP_s,   marker = 'D',     label = 'w/  Scheduler ' + ' {} lr_{}'.format(optimizer, lr))
#
#             # plt.plot(prob_arr, result_ICASSP_s,     marker = 'D',   label = 'ICASSP21 -'+ ' {} lr_{}'.format(optimizer, lr))
#             # plt.plot(prob_arr, result_ICML,         marker = '*',   label = 'ICML20- '  + ' {} lr_{}'.format(optimizer, lr))
#             # plt.plot(prob_arr, result_ICLR,         marker = 'o',   label = 'ICLR22- '  + ' {} lr_{}'.format(optimizer, lr))
#
#             # ax.plot(prob_arr, result_ICASSP_s,     marker = 'D',   label = 'ICASSP21')
#             # ax.plot(prob_arr, result_ICML,         marker = '*',   label = 'ICML20')
#             # ax.plot(prob_arr, result_ICLR,         marker = 'o',   label = 'ICLR22')
#             ax.plot(prob_arr, result_ICLR_mixup,   marker = '^',   label = 'ICLR22_mixup ' + ' {}'.format(wt))
#             ax.legend()
#             # plt.plot(prob_arr, result_ICLR_mixup,   marker = 'D', label = 'ICLR22_mixup- '+'optimizer_{} lr_{}'.format(optimizer, lr))
#             # color_idx += 1
#         counter+=1
#
#
# # for shuffle in ['True', 'False']:
# #     result_ICASSP_new=[]
# #     for prob in prob_arr:
# #         result_path = np.loadtxt(os.path.join(directory, 'ICASSP21/shuffle_{}/optimizer_{}/lr_{}/prob_{}.csv'.format(shuffle, 'adam', 0.1, prob)), delimiter=",")
# #         result_ICASSP_new.append(np.average(result_path))
# #
# #     plt.plot(prob_arr, result_ICASSP_new,   marker = 'D', label = 'ICASSP21-New-'+ 'shuffle_{} optimizer_{} lr_{}'.format(shuffle, 'adam', 0.1))
# #
# fig.suptitle('SEED_V: ICASSP21 Impact of Scheduler', fontsize=16)
#
# plt.legend()
# plt.show()
# exit(0)

# directory = '/home/patrick/Desktop/SEED_V_result/PLL/ICML22/optimizer_{}/lr_{}/cweight_{}_cweightw_{}_cweights_{}/'.format(args.optimizer, args.lr, args.c_weight, args.c_weight_w, args.c_weight_s)



# optimizer_arr = ['sgd']
# lr_arr = [0.1, 0.01, 0.001]
# wt_arr, wt_w_arr, wt_s_arr = [0.0,1.0], [0.0,1.0], [0.0,1.0]
# prob_arr = [0.2, 0.4, 0.6, 0.8]
#
#
#
# counter=0
# for optimizer in optimizer_arr:
#     for lr in lr_arr:
#         ax = fig.add_subplot(1, 3, counter+1)
#
#         for wt in wt_arr:
#             for wt_w in wt_w_arr:
#                 for wt_s in wt_s_arr:
#                     result_ICML=[]
#                     for prob in prob_arr:
#
#                         result_path_ICML22    = np.loadtxt(os.path.join(directory, 'ICML22/optimizer_{}/lr_{}/cweight_{}_cweightw_{}_cweights_{}/prob_{}.csv'.format(optimizer, lr, wt, wt_w, wt_s, prob)), delimiter=",")
#                         result_ICML.append(np.average(result_path_ICML22))
#                     ax.plot(prob_arr, result_ICML,    marker = '*', label = 'ICML22-'+ 'cweight_{} cweightw_{} cweights_{}'.format(wt, wt_w, wt_s))
#                     ax.legend()
#
#         counter+=1
#         ax.title.set_text('{} lr={}'.format(optimizer, lr))
#
#
# fig.suptitle('SEED_V: ICML22 Impact of weight applied on consistency loss', fontsize=16)
# plt.show()
#
#

#
# optimizer_arr = ['sgd']
# lr_arr = [0.1, 0.01, 0.001]
# wt_arr = ['warm_up', 'weight/0.5', 'weight/1', 'weight/2.0', 'weight/5.0']
# # wt_arr = ['weight/1']
#
# prob_arr = [0.2, 0.4, 0.6, 0.8]
#
def get_result(path):
    results = np.zeros((len(prob_arr), 5))
    for i in range(len(prob_arr)):
        for j in range(1, 6):
            results[i,j-1] = np.average(np.loadtxt(path.format(j,  prob_arr[i]), delimiter=','))
    return np.mean(results,1)

#
# #
# counter=0
# for optimizer in optimizer_arr:
#     for lr in lr_arr:
#         ax = fig.add_subplot(1, 3, counter+1)
#         result_ICML20=[]
#
#         for wt in wt_arr:
#
#             result_ICLR22_mixup=[]
#
#             result_ICASSP21_retest      = get_result(os.path.join(directory, 'ICASSP21_retest/optimizer_{}/lr_{}/'.format(optimizer, lr)) + 'run_{}/prob_{}.csv')
#             result_ICASSP21_retest_pair = get_result(os.path.join(directory, 'ICASSP21_retest/pair/optimizer_{}/lr_{}/'.format(optimizer, lr)) + 'run_{}/prob_{}.csv')
#
#             # for prob in prob_arr:
#
#                 # result_path_ICLR22_mixup        = np.loadtxt(os.path.join(directory, 'ICLR22_mixup/scheduler_True/optimizer_{}/lr_{}/{}/prob_{}.csv'.format(optimizer, lr, wt, prob)), delimiter=",")
#
#                 # result_path_ICML20    = np.loadtxt(os.path.join(directory, 'ICML20/optimizer_{}/lr_{}/prob_{}.csv'.format(optimizer, lr, prob)), delimiter=",")
#                 #
#                 # result_path_ICML20_retest              = np.loadtxt(os.path.join(directory, 'ICML20/shuffle_True/optimizer_{}/lr_{}/prob_{}.csv'.format(optimizer, lr, prob)), delimiter=",")
#
#                 # result_ICML20.append(np.average(result_path_ICML20))
#                 # result_ICLR22_mixup.append(np.average(result_path_ICLR22_mixup))
#
#                 # if wt==wt_arr[0]:
#
#             # ax.plot(prob_arr, result_ICLR22_mixup,               marker = '*', label = 'ICML20-')
#             if wt==wt_arr[0]:
#
#                 ax.plot(prob_arr, result_ICASSP21_retest,        marker = '*', color = 'black', label = 'ICASSP21-retest')
#                 ax.plot(prob_arr, result_ICASSP21_retest_pair,   marker = 'D', color = 'red', label = 'ICASSP21-retest_pair')
#             #
#             # print(result_ICASSP21_mixup)
#             # ax.plot(prob_arr, result_ICASSP21_mixup,         marker = 'D', label = 'ICASSP21-mixup' + '{}'.format(wt))
#
#
#         # ax.plot(prob_arr, result_ICML20,               marker = '*', color ='black', label = 'ICML20-')
#         # ax.plot(prob_arr, result_ICML20_retest,        marker = 'D', label = 'ICML20-retest-')
#
#         ax.legend()
#
#         counter+=1
#         ax.title.set_text('{} lr={}'.format(optimizer, lr))
#
#
# fig.suptitle('SEED_V: ICML22 Impact of weight applied on consistency loss', fontsize=16)
# plt.show()


# optimizer_arr = ['sgd']
# lr_arr = [0.1, 0.01, 0.001]
# # wt_arr = ['warm_up', 'weight/0.5', 'weight/1', 'weight/2.0', 'weight/5.0']
# # wt_arr = ['weight/1']
# init_arr = ['zero_init_True', 'zero_init_False']
# lw_arr  = [['0.0', '1.0'], ['0.0', '2.0'], ['1.0', '0.0'], ['1.0', '1.0'], ['1.0', '2.0'], ['2.0', '0.0'], ['2.0', '1.0'], ['2.0', '2.0']]
# prob_arr = [0.2, 0.4, 0.6, 0.8]
#
#
#
#
# for optimizer in optimizer_arr:
#     i =0
#     for lr in lr_arr:
#         j=0
#         for init in init_arr:
#             ax = fig.add_subplot(2, 3, (i+1) if j==0 else (i+1)+3)
#
#             for lw in lw_arr:
#                 # result_ICLR=[]
#                 # result_ICLR_mixup=[]
#                 result_ICML21=[]
#
#                 for prob in prob_arr:
#
#                     result_path_ICML21            = np.loadtxt(os.path.join(directory, 'ICML21/lr_{}/{}/lw_{}_lw0_{}/prob_{}.csv'.format(lr, init, lw[0], lw[1], prob)), delimiter=",")
#
#                     result_ICML21.append(np.average(result_path_ICML21))
#
#                 ax.plot(prob_arr, result_ICML21,               marker = '*',  label = 'lw={}, lw0={}'.format(lw[0], lw[1]))
#             ax.title.set_text('lr={} '.format(lr) + ' {}'.format(init))
#
#             ax.legend()
#
#             j+=1
#         i+=1
#
#
#
# fig.suptitle('SEED_V: ICML21 Impact of Confidence Initialization Method and Weight Applied on Partial and Non-Partial Loss', fontsize=16)
# plt.show()
#
#
#


# wt_arr = [0.0, 0.2, 0.4, 0.6, 0.8]
# epoch_arr = [0, 5, 10, 15, 20, 25]
# prob_arr = [0.2, 0.4, 0.6, 0.8]
#
# i=0
# for wt in wt_arr:
#     # j=0
#
#     ax = fig.add_subplot(1, 5, i+1)
#     for epoch in epoch_arr:
#
#         result_PiCO=[]
#         result_ICASSP=[]
#
#         for prob in prob_arr:
#             result_path_ICASSP21        = np.loadtxt(os.path.join(directory, 'ICASSP21_retest/optimizer_sgd/lr_{}/prob_{}.csv'.format(0.01, prob)), delimiter=",")
#
#             result_path_PiCO            = np.loadtxt(os.path.join(directory, 'PiCO/lr_{}/lossweight_{}/startepoch_{}/prob_{}.csv'.format(0.01, wt, epoch, prob)), delimiter=",")
#             # result_path_PiCO_scheduler  = np.loadtxt(os.path.join(directory, 'PiCO/scheduler/lr_{}/lossweight_{}/startepoch_{}/prob_{}.csv'.format(0.01, wt, epoch, prob)), delimiter=",")
#
#             result_PiCO.append(np.average(result_path_PiCO))
#             if epoch==epoch_arr[0]:
#                 result_ICASSP.append(np.average(result_path_ICASSP21))
#
#         # ax.plot(prob_arr, result_PiCO,               marker = '*',  label = 'proto start epoch={} '.format(epoch) + ' w/o scheduler')
#         if epoch==epoch_arr[0]:
#             ax.plot(prob_arr, result_ICASSP,   marker = '*',  color='black', label = 'ICASSP21')
#         ax.plot(prob_arr, result_PiCO,     marker = 'D',  label = 'proto start epoch={} '.format(epoch))
#
#
#     ax.title.set_text('lossweight={} '.format(wt))
#
#     ax.legend()
#
#         # j+=1
#     i+=1
#
#
# #
# fig.suptitle('SEED_V: PiCO w/o Scheduler, Impact of Constrastive Loss Weight and Prototype Start Epoch', fontsize=16)
# plt.show()


sche_arr = ['scheduler_True']
init_arr = ['zero_init_True', 'zero_init_False']
part_arr = ['binomial']

thres_arr = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
prob_arr = [0.2, 0.4, 0.6, 0.8, 0.9, 0.9]

i=0
for part in part_arr:
    for init in init_arr:
        ax = fig.add_subplot(2, 2, i+1)
        i+=1
        for thres in thres_arr:

            for prob in prob_arr:
                result_ICASSP21_retest  = get_result(os.path.join(directory, 'ICASSP21_retest/{}/optimizer_{}/lr_{}/'.format(part, 'sgd', 0.01)) + 'run_{}/prob_{}.csv')
                # result_path_ours            = np.loadtxt(os.path.join(directory, 'ICASSP21_ema_confidence/lr_{}/{}/threshold_{}/prob_{}.csv'.format(0.01, init, thres, prob)), delimiter=",")
                result_path_ours  = get_result(os.path.join(directory, 'ICASSP21_confidence/{}/{}/{}/threshold_{}/'.format('scheduler_True', part, init, thres)) + 'run_{}/prob_{}.csv')

                result_path_ours_auto  = get_result(os.path.join(directory, 'ICASSP21_auto_confidence/{}/{}/{}/'.format('scheduler_True', part, init)) + 'run_{}/prob_{}.csv')


            if thres==thres_arr[0]:
                ax.plot(prob_arr, result_ICASSP21_retest,       marker = '*', color = 'black',  label =  'ICASSP21')
                ax.plot(prob_arr, result_path_ours_auto,        marker = '*', color = 'red',    label =  'ICASSP21-auto')

                # ax.plot(prob_arr, result_ICASSP21_retest_pair,  marker = '^', color = 'black', label =  'ICASSP21-pair')


            ax.plot(prob_arr, result_path_ours,              marker = '*',  label = 'threshold={}'.format(thres))

            # ax.plot(prob_arr, result_ours_scheduler,    marker = 'D',  label = 'threshold={}'.format(thres))



        ax.title.set_text('Partial Label Generation={} '.format(part) + 'Conf-Init={} '.format(init))

        ax.legend()



fig.suptitle('SEED_V (Binomial partial label gneration): Ours(ICASSP21 + confidence threshold + with Scheduler) Vs. ICASSP21', fontsize=16)
plt.show()


#











#
