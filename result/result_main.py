import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
# directory = '/home/patrick/Desktop/SEED_V_result/PLL/main/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/cross_entropy/beta_2.0/run_{}/prob_{}.csv'

directory = '/home/patrick/Desktop/SEED_V_result/PLL/main/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/run_{}/prob_{}.csv'

#
#
#
arr = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
def get_result(path):
    results_mean = np.zeros((len(arr), 5))
    results_std  = np.zeros((len(arr), 5))

    for i in range(len(arr)):
        for j in range(1, 6):
            results_mean[i,j-1] = np.average(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
            results_std[i,j-1]  = np.std(np.loadtxt(path.format(j,  arr[i]), delimiter=','))

    return np.mean(results_mean,1), np.mean(results_std,1)



print(get_result(directory)[0])



# directory = '/home/patrick/Desktop/SEED_V_result/PLL/main/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/cross_entropy/beta_2.0/run_{}/gamma.csv'
directory = '/home/patrick/Desktop/SEED_V_result/PLL/main/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/run_{}/gamma.csv'

# directory = '/home/patrick/Desktop/SEED_V_result/PLL/main/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/run_{}/gamma.csv'


def get_result(path):
    results_mean = np.zeros((5,1))
    results_std  = np.zeros((5,1))


    for i in range(1, 6):
        results_mean[i-1] = np.average(np.loadtxt(path.format(i), delimiter=','))
        results_std[i-1]  = np.std(np.loadtxt(path.format(i), delimiter=','))

    return np.mean(results_mean,0), np.mean(results_std,0)

print(get_result(directory)[0])
#
# directory = '/home/patrick/Desktop/SEED_V_result/temp/PLL/ICML21/LWC/lw_2.0_lw0_1.0/run_{}/prob_{}.csv'
#
#
# arr = [0.8, 0.9, 0.95]
# def get_result(path):
#     results_mean = np.zeros((len(arr), 5))
#     results_std  = np.zeros((len(arr), 5))
#
#     for i in range(len(arr)):
#         for j in range(1, 6):
#             results_mean[i,j-1] = np.average(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
#             results_std[i,j-1]  = np.std(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
#
#     return np.mean(results_mean,1), np.mean(results_std,1)
#
#
#
# print(get_result(directory)[0])
