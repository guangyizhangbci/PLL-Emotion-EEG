import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

directory = '/home/patrick/Desktop/SEED_V_result/PLL/'




arr = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
def get_result(path):
    results_mean = np.zeros((len(arr), 5))
    results_std  = np.zeros((len(arr), 5))

    for i in range(len(arr)):
        for j in range(1, 6):
            results_mean[i,j-1] = np.average(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
            results_std[i,j-1]  = np.std(np.loadtxt(path.format(j,  arr[i]), delimiter=','))

    return np.mean(results_mean,1), np.mean(results_std,1)


def split_float(x):
    before, after = str(x).split('.')
    return int(before)

def make_num(mean, std, best_flag, second_flag):
    mean, std = mean*100, std*100
    if  best_flag ==True:
        if split_float(std) <10:
            output = '\\textbf{' + '{0:.2f} '.format(np.round(mean,4))+ '$ $\\tiny' + '({0:.2f})'.format(np.round(std,4)) + '}'
        else:
            output = '\\textbf{' + '{0:.2f}'.format(np.round(mean,4))+ '\\tiny'+ '({0:.2f})'.format(np.round(std,4)) + '}'

    elif  second_flag ==True:
        if split_float(std) <10:
            output = '\\underline{' + '{0:.2f} '.format(np.round(mean,4))+ '$ $\\tiny' + '({0:.2f})'.format(np.round(std,4)) + '}'
        else:
            output = '\\underline{' + '{0:.2f}'.format(np.round(mean,4))+ '\\tiny' + '({0:.2f})'.format(np.round(std,4)) + '}'

    else:
        if split_float(std) <10:
            output = '{0:.2f} '.format(np.round(mean,4))+ '$ $\\tiny' +  '({0:.2f})'.format(np.round(std,4))
        else:
            output = '{0:.2f}'.format(np.round(mean,4))+ '\\tiny' + '({0:.2f})'.format(np.round(std,4))

    # output =  str(np.round(mean*100,2)) + ' $\\pm$ ' + str(np.round(std*100,2))
    return output


# ################################
lw_arr  = ['0.0', '1.0', '2.0']
#
#
#
# for lw in lw_arr:
#
#     result_ICML21_lws  = get_result(os.path.join(directory, 'ICML21/LWS/lw_{}_lw0_1.0/'.format(lw) + 'run_{}/prob_{}.csv'))
#     result_ICML21_lwc  = get_result(os.path.join(directory, 'ICML21/LWC/lw_{}_lw0_1.0/'.format(lw) + 'run_{}/prob_{}.csv'))
#
#     result_ICML21_lws_slo  = get_result(os.path.join(directory, 'ICML21/LWS/supervised_loss_only/lw_{}_lw0_1.0/'.format(lw) + 'run_{}/prob_{}.csv'))
#     result_ICML21_lwc_slo  = get_result(os.path.join(directory, 'ICML21/LWC/supervised_loss_only/lw_{}_lw0_1.0/'.format(lw) + 'run_{}/prob_{}.csv'))
#
#
#     # plt.plot(arr, result_ICML21_lws[0],        marker = 'D',  label = 'LWS, lw={}'.format(lw[0]))
#
#     plt.plot(arr, result_ICML21_lwc[0],        marker = 'D',  label = 'LWC, lw={}'.format(lw[0]))
#
#     # plt.plot(arr, result_ICML21_lws_slo[0],    marker = '*',  label = 'LWS-supervised_loss_only, lw={}'.format(lw[0]))
#     plt.plot(arr, result_ICML21_lwc_slo[0],    marker = '*',  label = 'LWC-supervised_loss_only, lw={}'.format(lw[0]))
#
#
# plt.title('ICML21, W/ label disambigution')
# plt.xlabel('Label Ambiguity')
# plt.legend()
# plt.show()


lw_arr  = ['0.0', '1.0', '2.0']
name_arr = ['sigmoid/', 'cross_entropy/']

result_mean = np.zeros((0, 6))
result_std  = np.zeros((0, 6))

dirname = directory + 'main/LW/scheduler_True/optimizer_sgd/lr_0.01/'

for name in name_arr:
    for lw in lw_arr:
        beta = 'beta_{}/'.format(lw)
        wo_LD = get_result(dirname +'confidence_False/' + name + 'beta_{}/'.format(lw) + 'run_{}/prob_{}.csv')
        w_LD  = get_result(dirname +'confidence_True/'  + name + 'beta_{}/'.format(lw) + 'run_{}/prob_{}.csv')


        result_mean = np.vstack((result_mean, wo_LD[0], w_LD[0]))
        result_std  = np.vstack((result_std,  wo_LD[1], w_LD[1]))


#
# result_mean = np.zeros((0, 6))
# result_std  = np.zeros((0, 6))
#
# dir_name = directory + 'PiCO/proto_start_{}/contrast_weight_{}/'
#
# wo_proto_wo_contra = get_result(dir_name.format(30, 0.0) + 'run_{}/prob_{}.csv')
# wo_proto_w_contra  = get_result(dir_name.format(30, 0.5) + 'run_{}/prob_{}.csv')
# w_proto_wo_contra  = get_result(dir_name.format(0, 0.0)  + 'run_{}/prob_{}.csv')
# w_proto_w_contra   = get_result(dir_name.format(0, 0.5)  + 'run_{}/prob_{}.csv')
#
# result_mean = np.vstack((wo_proto_wo_contra[0], wo_proto_w_contra[0], w_proto_wo_contra[0], w_proto_w_contra[0]))
# result_std  = np.vstack((wo_proto_wo_contra[1], wo_proto_w_contra[1], w_proto_wo_contra[1], w_proto_w_contra[1]))
#
#
#

#
#
# name_arr  = ['Naive',    'PRODEN',  'CAVL',   'LW',     'CR',     'PiCO']
# venue_arr = ['ICASSP21', 'ICML20',  'ICLR22', 'ICML21', 'ICML22', 'ICLR22']


naive_result        = get_result(directory + 'main/DNPL/scheduler_False/optimizer_sgd/lr_0.01/' + 'run_{}/prob_{}.csv')

proden_result_wo    = get_result(directory + 'main/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/' + 'run_{}/prob_{}.csv')
proden_result_w     = get_result(directory + 'main/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'  + 'run_{}/prob_{}.csv')

cavl_result_wo      = get_result(directory + 'main/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'   + 'run_{}/prob_{}.csv')
cavl_result_w       = get_result(directory + 'main/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'    + 'run_{}/prob_{}.csv')

lw_result_wo        = get_result(directory + 'main/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/cross_entropy/beta_2.0/'     + 'run_{}/prob_{}.csv')
lw_result_w         = get_result(directory + 'main/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/cross_entropy/beta_2.0/'      + 'run_{}/prob_{}.csv')

# cr_result_wo     =  get_result(directory + 'ICML22/' + 'supervised_loss_only/'+ 'run_{}/prob_{}.csv')
cr_result_wo        = get_result(directory + 'main/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/'     + 'run_{}/prob_{}.csv')
cr_result_w         = get_result(directory + 'main/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/weight_1.0_1.0_1.0/'     + 'run_{}/prob_{}.csv')

wo_proto_wo_contra  = get_result(directory + 'main/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.0/'    + 'run_{}/prob_{}.csv')
w_proto_wo_contra   = get_result(directory + 'main/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.0/'    + 'run_{}/prob_{}.csv')
wo_proto_w_contra   = get_result(directory + 'main/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.5/'    + 'run_{}/prob_{}.csv')
w_proto_w_contra    = get_result(directory + 'main/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.5/'    + 'run_{}/prob_{}.csv')




result_mean = np.vstack((naive_result[0], proden_result_wo[0], proden_result_w[0], cavl_result_wo[0], cavl_result_w[0], lw_result_wo[0], lw_result_w[0], cr_result_wo[0], cr_result_w[0], wo_proto_w_contra[0], w_proto_w_contra[0]))
result_std  = np.vstack((naive_result[1], proden_result_wo[1], proden_result_w[1], cavl_result_wo[1], cavl_result_w[1], lw_result_wo[1], lw_result_w[1], cr_result_wo[1], cr_result_w[1], wo_proto_w_contra[1], w_proto_w_contra[1]))




# result_mean = np.vstack((wo_proto_wo_contra[0], w_proto_wo_contra[0], wo_proto_w_contra[0], w_proto_w_contra[0]))
# result_std  = np.vstack((wo_proto_wo_contra[1], w_proto_wo_contra[1], wo_proto_w_contra[1], w_proto_w_contra[1]))

# print(np.mean(result_mean[0])- np.mean(result_mean[6]))
# exit(0)
# #
# for beta_va



keys = ['DNPL', 'PRODEN', 'PRODEN', 'CAVL', 'CAVL', 'LW', 'LW', 'CR', 'CR', 'PiCO', 'PiCO']

index = [i for i in range(len(keys))]

values = np.mean(result_mean, axis=1)
#
plt.rcParams['axes.xmargin'] = 0.02
# fig, ax = plt.subplots()
fig= plt.figure()
result_mean = np.mean(result_mean, axis=1)

result_wo_ld = [result_mean[0]] + [result_mean[i] for i in range(1,len(result_mean)) if i%2==1]
result_w_ld  = [None] + [result_mean[i] for i in range(1,len(result_mean)) if i%2==0]



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


best   = np.argsort(result_mean, axis=0)[-1]
second = np.argsort(result_mean, axis=0)[-2]


new_result = ''
best_flag, second_flag = False, False

for i in range(result_mean.shape[0]):

    # print(name)
    for j in range(6):
        if  best[j]==i:
            best_flag = True
        else:
            best_flag = False
        if  second[j]==i:
            second_flag = True
        else:
            second_flag = False
        # print(i,j,best_flag, second_flag)

        # if  j==0:
        #     new_result = name + '\t&\t' + make_num(result_mean[i,j], result_std[i,j], best_flag=best_flag, second_flag=second_flag)
        # else:
        new_result += make_num(result_mean[i,j], result_std[i,j], best_flag=best_flag, second_flag=second_flag)

        if  j < 5:
            new_result += '\t&\t'

    if  i < result_mean.shape[0]:
        new_result += '\t\\\\\t\n'
print(new_result)




























#
