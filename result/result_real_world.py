import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

directory = '/home/patrick/Desktop/SEED_V_result/PLL/'




arr = [0]
def get_result(path):
    results_mean = np.zeros((len(arr), 5))
    results_std  = np.zeros((len(arr), 5))

    for i in range(len(arr)):
        for j in range(1, 6):
            results_mean[i,j-1] = np.average(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
            results_std[i,j-1]  = np.std(np.loadtxt(path.format(j,  arr[i]), delimiter=','))

    return np.mean(results_mean,1), np.mean(results_std,1)




# ################################
lw_arr  = ['0.0', '1.0', '2.0']
#

#
#
name_arr  = ['Naive',    'PRODEN',  'CAVL',   'LW',     'CR',     'PiCO']
venue_arr = ['ICASSP21', 'ICML20',  'ICLR22', 'ICML21', 'ICML22', 'ICLR22']


naive_result     =  get_result(directory + 'main/emotion_delta/1.0/DNPL/scheduler_False/optimizer_sgd/lr_0.01/' + 'run_{}/gamma.csv')

proden_result_wo =  get_result(directory + 'main/emotion_delta/1.0/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/' + 'run_{}/gamma.csv')
proden_result_w  =  get_result(directory + 'main/emotion_delta/1.0/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'  + 'run_{}/gamma.csv')

cavl_result_wo   =  get_result(directory + 'main/emotion_delta/1.0/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'   + 'run_{}/gamma.csv')
cavl_result_w    =  get_result(directory + 'main/emotion_delta/1.0/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'    + 'run_{}/gamma.csv')

lw_result_wo     =  get_result(directory + 'main/emotion_delta/1.0/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/cross_entropy/beta_2.0/'     + 'run_{}/gamma.csv')
lw_result_w      =  get_result(directory + 'main/emotion_delta/1.0/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/cross_entropy/beta_2.0/'      + 'run_{}/gamma.csv')

# cr_result_wo     =  get_result(directory + 'ICML22/' + 'supervised_loss_only/'+ 'run_{}/gamma.csv')
cr_result_wo     =  get_result(directory + 'main/emotion_delta/1.0/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/'     + 'run_{}/gamma.csv')
cr_result_w      =  get_result(directory + 'main/emotion_delta/1.0/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/weight_1.0_1.0_1.0/'      + 'run_{}/gamma.csv')


wo_proto_wo_contra   = get_result(directory + 'main/emotion_delta/1.0/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.0/'    + 'run_{}/gamma.csv')
w_proto_wo_contra    = get_result(directory + 'main/emotion_delta/1.0/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.0/'     + 'run_{}/gamma.csv')
wo_proto_w_contra    = get_result(directory + 'main/emotion_delta/1.0/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.5/'    + 'run_{}/gamma.csv')
w_proto_w_contra     = get_result(directory + 'main/emotion_delta/1.0/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.5/'     + 'run_{}/gamma.csv')



result_mean = np.vstack((naive_result[0], proden_result_wo[0], proden_result_w[0], cavl_result_wo[0], cavl_result_w[0], lw_result_wo[0], lw_result_w[0], cr_result_wo[0], cr_result_w[0], wo_proto_w_contra[0], w_proto_w_contra[0]))
result_std  = np.vstack((naive_result[1], proden_result_wo[1], proden_result_w[1], cavl_result_wo[1], cavl_result_w[1], lw_result_wo[1], lw_result_w[1], cr_result_wo[1], cr_result_w[1], wo_proto_w_contra[1], w_proto_w_contra[1]))

# print(result_mean)
# exit(0)

keys = ['DNPL', 'PRODEN', 'PRODEN', 'CAVL', 'CAVL', 'LW', 'LW', 'CR', 'CR', 'PiCO', 'PiCO']

index = [i for i in range(len(keys))]

values = [result_mean[i,0] for i in range(len(result_mean))]

'''
print((values))
plt.rcParams['axes.xmargin'] = 0.02
# fig, ax = plt.subplots()
fig= plt.figure()

a, b = keys, values
# c=['gray', 'lightseagreen', 'brown', 'royalblue']
plt.bar(index,b, width = 0.5, color=['tab:blue', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:orange'])
plt.xticks(index, a, rotation = 45, ha="right")

colors = {'w/o LD':'tab:blue','w/ LD':'tab:orange'}
labels = list(colors.keys())

handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)

import matplotlib.ticker as ticker
# fig.tight_layout()
plt.ylabel('Acc.')

# ax.set_xticks(np.arange(11), list(mean_result.keys()))
# ax.set_xticklabels(list(mean_result.keys()), rotation = 45, ha="right")

# ax.set_ylim([min(list(mean_result.values()))-0.01, max(list(mean_result.values()))+0.01])
plt.grid(axis='y')
#
plt.show()

'''
plt.rcParams['axes.xmargin'] = 0.02
# fig, ax = plt.subplots()
fig= plt.figure()

result_wo_ld = [result_mean[0,0]] + [result_mean[i,0] for i in range(1,len(result_mean)) if i%2==1]
result_w_ld  = [None] + [result_mean[i,0] for i in range(1,len(result_mean)) if i%2==0]



xs = np.arange(6)
series1 = np.array(result_wo_ld).astype(np.double)
s1mask = np.isfinite(series1)
series2 = np.array(result_w_ld).astype(np.double)
s2mask = np.isfinite(series2)

# plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o',  markersize=8, label ='w/o LD')
# plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker='o',  markersize=8, label ='w/'+'   '+'LD')

bar_width = 0.3

# Create the bar chart
plt.bar(xs[s1mask] - bar_width/2, series1[s1mask], width=bar_width, label ='w/o LD')
plt.bar(xs[s2mask] + bar_width/2, series2[s2mask], width=bar_width, label ='w/'+'   '+'LD')

# Find min and max values from both series for y-axis scaling
min_value = min(np.min(series1[s1mask]), np.min(series2[s2mask]))
max_value = max(np.max(series1[s1mask]), np.max(series2[s2mask]))

# Adjust the y-axis limits (min_value - 0.05 to max_value + 0.05)
plt.ylim(min_value - 0.01, max_value + 0.01)

plt.ylabel('Acc.', fontsize=12)
plt.xticks([0,1,2,3,4,5], ['DNPL' , 'PRODEN', 'CAVL', 'LW','CR', 'PiCO'])
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()


exit(0)



# best   = np.argsort(result_mean, axis=0)[-1]
# second = np.argsort(result_mean, axis=0)[-2]
#
#
# new_result = ''
# best_flag, second_flag = False, False
#
# for i in range(result_mean.shape[0]):
#
#     # print(name)
#     for j in range(6):
#         if  best[j]==i:
#             best_flag = True
#         else:
#             best_flag = False
#         if  second[j]==i:
#             second_flag = True
#         else:
#             second_flag = False
#         # print(i,j,best_flag, second_flag)
#
#         # if  j==0:
#         #     new_result = name + '\t&\t' + make_num(result_mean[i,j], result_std[i,j], best_flag=best_flag, second_flag=second_flag)
#         # else:
#         new_result += make_num(result_mean[i,j], result_std[i,j], best_flag=best_flag, second_flag=second_flag)
#
#         if  j < 5:
#             new_result += '\t&\t'
#
#     if  i < result_mean.shape[0]:
#         new_result += '\t\\\\\t\n'
# print(new_result)
#
#
#
#
























#
