import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

directory = '/home/patrick/Desktop/SEED_V_result/PLL/main/'




arr = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
def get_result(path):
    results_mean = np.zeros((len(arr), 5))
    results_std  = np.zeros((len(arr), 5))

    for i in range(len(arr)):
        for j in range(1, 6):
            results_mean[i,j-1] = np.average(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
            results_std[i,j-1]  = np.std(np.loadtxt(path.format(j,  arr[i]), delimiter=','))

    return np.mean(results_mean,1), np.mean(results_std,1)



fig= plt.figure()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 8})




result_ICASSP21 = get_result(os.path.join(directory, 'ICASSP/run_{}/prob_{}.csv'))
result_Simple_CE = get_result(os.path.join(directory, 'Simple_CE/run_{}/prob_{}.csv'))

result_ICML22_SL  = get_result(os.path.join(directory, 'ICML22/supervised_loss_only/run_{}/prob_{}.csv'))


plt.plot(arr, result_ICASSP21[0],  marker = 'D', label='ICASSP')
plt.plot(arr, result_Simple_CE[0], marker = '*', label='Simple')
# plt.plot(result_ICLR22_SL, label='ICML21 SL Only')
plt.plot(arr, result_ICML22_SL[0], marker = '*', label='ICML22 w/o disambigution')

plt.xlabel('Label Ambiguity')
plt.grid()

plt.legend()
plt.show()









################################
lw_arr  = [['0.0', '1.0'], ['1.0', '1.0'], ['2.0', '1.0']]



for lw in lw_arr:

    result_ICML21_lws  = get_result(os.path.join(directory, 'ICML21/LWS/lw_{}_lw0_{}/'.format(lw[0], lw[1]) + 'run_{}/prob_{}.csv'))
    result_ICML21_lwc  = get_result(os.path.join(directory, 'ICML21/LWC/lw_{}_lw0_{}/'.format(lw[0], lw[1]) + 'run_{}/prob_{}.csv'))

    result_ICML21_lws_slo  = get_result(os.path.join(directory, 'ICML21/LWS/supervised_loss_only/lw_{}_lw0_{}/'.format(lw[0], lw[1]) + 'run_{}/prob_{}.csv'))
    result_ICML21_lwc_slo  = get_result(os.path.join(directory, 'ICML21/LWC/supervised_loss_only/lw_{}_lw0_{}/'.format(lw[0], lw[1]) + 'run_{}/prob_{}.csv'))


    # plt.plot(arr, result_ICML21_lws[0],        marker = 'D',  label = 'LWS, lw={}'.format(lw[0]))

    plt.plot(arr, result_ICML21_lwc[0],        marker = 'D',  label = 'LWC, lw={}'.format(lw[0]))

    # plt.plot(arr, result_ICML21_lws_slo[0],    marker = '*',  label = 'LWS-supervised_loss_only, lw={}'.format(lw[0]))
    plt.plot(arr, result_ICML21_lwc_slo[0],    marker = '*',  label = 'LWC-supervised_loss_only, lw={}'.format(lw[0]))


plt.title('ICML21, W/ label disambigution')
plt.xlabel('Label Ambiguity')
plt.legend()
plt.show()











#
