import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# directory = '/home/patrick/Desktop/SEED_IV_result/PLL/'

directory = '/home/patrick/Desktop/SEED_V_result/PLL/main/emotion_delta/'



arr = [0]
def get_result(path):
    results_mean = np.zeros((len(arr), 5))
    results_std  = np.zeros((len(arr), 5))

    for i in range(len(arr)):
        for j in range(1, 6):
            results_mean[i,j-1] = np.average(np.loadtxt(path.format(j,  arr[i]), delimiter=','))
            results_std[i,j-1]  = np.std(np.loadtxt(path.format(j,  arr[i]), delimiter=','))

    return np.mean(results_mean,1), np.mean(results_std,1)


import glob

def get_prob_file_path(directory, delta, subfolder):
    # Construct the base directory
    base_dir = directory + '{}/'.format(delta) + subfolder + 'run_1/'
    
    # Dynamically search for the prob_*.csv file in the folder
    prob_file_pattern = os.path.join(base_dir, 'prob_*.csv')
    prob_files = glob.glob(prob_file_pattern)
    
    if len(prob_files) == 1:
        return prob_files[0]  # Return the full path of the found file
    else:
        raise FileNotFoundError(f"Expected 1 'prob_*.csv' file, but found {len(prob_files)} files in {base_dir}")


# ################################
lw_arr  = ['0.0', '1.0', '2.0']
#

#
#
name_arr  = ['Naive',    'PRODEN',  'CAVL',   'LW',     'CR',     'PiCO']
venue_arr = ['ICASSP21', 'ICML20',  'ICLR22', 'ICML21', 'ICML22', 'ICLR22']







import matplotlib.pyplot as plt
import numpy as np

# Prepare delta values and empty arrays to store results
delta_values = [1.0, 2.0, 3.0, 4.0, 5.0]

# Initialize lists to collect results
naive_results_real = []
naive_results_uniform = []

# Collect results for all methods
proden_results_wo_real, proden_results_w_real, proden_results_wo_uniform, proden_results_w_uniform = [], [], [], []
cavl_results_wo_real, cavl_results_w_real, cavl_results_wo_uniform, cavl_results_w_uniform = [], [], [], []
lw_results_wo_real, lw_results_w_real, lw_results_wo_uniform, lw_results_w_uniform = [], [], [], []
cr_results_wo_real, cr_results_w_real, cr_results_wo_uniform, cr_results_w_uniform = [], [], [], []
pico_results_wo_real, pico_results_w_real, pico_results_wo_uniform, pico_results_w_uniform = [], [], [], []

# Loop through each delta value and collect results
for delta in delta_values:

    # DNPL results
    naive_results_real.append(get_result(directory + '{}/DNPL/scheduler_False/optimizer_sgd/lr_0.01/'.format(delta) + 'run_{}/gamma.csv'))
    naive_results_uniform.append(get_result(get_prob_file_path(directory, delta, 'DNPL/scheduler_False/optimizer_sgd/lr_0.01/')))

    # PRODEN results
    proden_results_wo_real.append(get_result(directory + '{}/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'.format(delta) + 'run_{}/gamma.csv'))
    proden_results_w_real.append(get_result(directory + '{}/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'.format(delta) + 'run_{}/gamma.csv'))
    proden_results_wo_uniform.append(get_result(get_prob_file_path(directory, delta, 'PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/')))
    proden_results_w_uniform.append(get_result(get_prob_file_path(directory, delta, 'PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/')))

    # CAVL results
    cavl_results_wo_real.append(get_result(directory + '{}/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'.format(delta) + 'run_{}/gamma.csv'))
    cavl_results_w_real.append(get_result(directory + '{}/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'.format(delta) + 'run_{}/gamma.csv'))
    cavl_results_wo_uniform.append(get_result(get_prob_file_path(directory, delta, 'CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/')))
    cavl_results_w_uniform.append(get_result(get_prob_file_path(directory, delta, 'CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/')))

    # LW results
    lw_results_wo_real.append(get_result(directory + '{}/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/cross_entropy/beta_2.0/'.format(delta) + 'run_{}/gamma.csv'))
    lw_results_w_real.append(get_result(directory + '{}/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/cross_entropy/beta_2.0/'.format(delta) + 'run_{}/gamma.csv'))
    lw_results_wo_uniform.append(get_result(get_prob_file_path(directory, delta, 'LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/cross_entropy/beta_2.0/')))
    lw_results_w_uniform.append(get_result(get_prob_file_path(directory, delta, 'LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/cross_entropy/beta_2.0/')))

    # CR results
    cr_results_wo_real.append(get_result(directory + '{}/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/'.format(delta) + 'run_{}/gamma.csv'))
    cr_results_w_real.append(get_result(directory + '{}/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/weight_1.0_1.0_1.0/'.format(delta) + 'run_{}/gamma.csv'))
    cr_results_wo_uniform.append(get_result(get_prob_file_path(directory, delta, 'CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/')))
    cr_results_w_uniform.append(get_result(get_prob_file_path(directory, delta, 'CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/weight_1.0_1.0_1.0/')))

    # PiCO results
    pico_results_wo_real.append(get_result(directory + '{}/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.5/'.format(delta) + 'run_{}/gamma.csv'))
    pico_results_w_real.append(get_result(directory + '{}/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.5/'.format(delta) + 'run_{}/gamma.csv'))
    pico_results_wo_uniform.append(get_result(get_prob_file_path(directory, delta, 'PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.5/')))
    pico_results_w_uniform.append(get_result(get_prob_file_path(directory, delta, 'PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.5/')))

# Plotting with markers
fig, axes = plt.subplots(2, 3, figsize=(12, 10))
axes = axes.flatten()

# Set markers for all plots
marker1 = 'D'
marker2 = 'o'


color1 ='#33a8ff'  # Classical   + w/o LD
color2 ='orange'   # Classical   + w/ LD
color3 ='#0072BD'  # Real-World  + w/o LD
color4 ='#ff6433'  # Real-World  + w/  LD


# Plot DNPL (2 curves)
axes[0].plot(delta_values, [res[0] for res in naive_results_uniform],   marker=marker1, color=color3,   label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/o LD')
axes[0].plot(delta_values, [res[0] for res in naive_results_real],      marker=marker2, color=color1,   label='Real-World ' + 'w/o LD')
axes[0].set_title('DNPL')


# Plot PRODEN (4 curves)
axes[1].plot(delta_values, [res[0] for res in proden_results_wo_uniform],   marker=marker1, color=color1,   label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/o LD')
axes[1].plot(delta_values, [res[0] for res in proden_results_w_uniform],    marker=marker1, color=color2,   label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/   LD')
axes[1].plot(delta_values, [res[0] for res in proden_results_wo_real],      marker=marker2, color=color3,   label='Real-World ' + 'w/o LD')
axes[1].plot(delta_values, [res[0] for res in proden_results_w_real],       marker=marker2, color=color4,   label='Real-World ' + 'w/   LD')
axes[1].set_title('PRODEN')

# Plot CAVL (4 curves)
axes[2].plot(delta_values, [res[0] for res in cavl_results_wo_uniform], marker=marker1, color=color1,   label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/o LD')
axes[2].plot(delta_values, [res[0] for res in cavl_results_w_uniform],  marker=marker1, color=color2,   label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/   LD')
axes[2].plot(delta_values, [res[0] for res in cavl_results_wo_real],    marker=marker2, color=color3,   label='Real-World ' + 'w/o LD')
axes[2].plot(delta_values, [res[0] for res in cavl_results_w_real],     marker=marker2, color=color4,   label='Real-World ' + 'w/   LD')
axes[2].set_title('CAVL')


# Plot LW (4 curves)
axes[3].plot(delta_values, [res[0] for res in lw_results_wo_uniform],   marker=marker1, color=color1,  label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/o LD')
axes[3].plot(delta_values, [res[0] for res in lw_results_w_uniform],    marker=marker1, color=color2,  label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/   LD')
axes[3].plot(delta_values, [res[0] for res in lw_results_wo_real],      marker=marker2, color=color3,  label='Real-World ' + 'w/o LD')
axes[3].plot(delta_values, [res[0] for res in lw_results_w_real],       marker=marker2, color=color4,  label='Real-World ' + 'w/   LD')
axes[3].set_title('LW')


# Plot CR (4 curves)
axes[4].plot(delta_values, [res[0] for res in cr_results_wo_uniform],   marker=marker1, color=color1,  label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/o LD')
axes[4].plot(delta_values, [res[0] for res in cr_results_w_uniform],    marker=marker1, color=color2,  label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/   LD')
axes[4].plot(delta_values, [res[0] for res in cr_results_wo_real],      marker=marker2, color=color3,  label='Real-World ' + 'w/o LD')
axes[4].plot(delta_values, [res[0] for res in cr_results_w_real],       marker=marker2, color=color4,  label='Real-World ' + 'w/   LD')
axes[4].set_title('CR')


# Plot PiCO (4 curves)
axes[5].plot(delta_values, [res[0] for res in pico_results_wo_uniform], marker=marker1, color=color1,  label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/o LD')
axes[5].plot(delta_values, [res[0] for res in pico_results_w_uniform],  marker=marker1, color=color2,  label='Classical\u00A0\u00A0\u00A0\u00A0' + 'w/   LD')
axes[5].plot(delta_values, [res[0] for res in pico_results_wo_real],    marker=marker2, color=color3,  label='Real-World ' + 'w/o LD')
axes[5].plot(delta_values, [res[0] for res in pico_results_w_real],     marker=marker2, color=color4,  label='Real-World ' + 'w/   LD')

axes[5].set_title('PiCO')

for i in range (0, 6):

    axes[i].set_xlabel('$\delta$',fontsize=11)
    axes[i].set_ylabel('Acc.', fontsize=11)
    axes[i].legend(loc='lower left', fontsize=8)
    axes[i].set_xticks(delta_values)
    axes[i].grid(True, axis='y') 

axes[4].legend(fontsize=8)
plt.tight_layout()
plt.show()



















# for delta in [2.0, 3.0, 4.0, 5.0]:

#     naive_result_real     =  get_result(directory + '{}/DNPL/scheduler_False/optimizer_sgd/lr_0.01/'.format(delta) + 'run_{}/gamma.csv')

#     proden_result_wo_real =  get_result(directory + '{}/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'.format(delta)  + 'run_{}/gamma.csv')
#     proden_result_w_real  =  get_result(directory + '{}/PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'.format(delta)   + 'run_{}/gamma.csv')

#     cavl_result_wo_real   =  get_result(directory + '{}/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'.format(delta)    + 'run_{}/gamma.csv')
#     cavl_result_w_real    =  get_result(directory + '{}/CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'.format(delta)     + 'run_{}/gamma.csv')

#     lw_result_wo_real     =  get_result(directory + '{}/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/cross_entropy/beta_2.0/'.format(delta)      + 'run_{}/gamma.csv')
#     lw_result_w_real      =  get_result(directory + '{}/LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/cross_entropy/beta_2.0/'.format(delta)       + 'run_{}/gamma.csv')

#     # cr_result_wo     =  get_result(directory + 'ICML22/' + 'supervised_loss_only/'+ 'run_{}/gamma.csv')
#     cr_result_wo_real     =  get_result(directory + '{}/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/'.format(delta)      + 'run_{}/gamma.csv')
#     cr_result_w_real      =  get_result(directory + '{}/CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/weight_1.0_1.0_1.0/'.format(delta)       + 'run_{}/gamma.csv')


#     wo_proto_wo_contra_real   = get_result(directory + '{}/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.0/'.format(delta)     + 'run_{}/gamma.csv')
#     w_proto_wo_contra_real    = get_result(directory + '{}/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.0/'.format(delta)      + 'run_{}/gamma.csv')
#     wo_proto_w_contra_real    = get_result(directory + '{}/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.5/'.format(delta)     + 'run_{}/gamma.csv')
#     w_proto_w_contra_real     = get_result(directory + '{}/PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.5/'.format(delta)      + 'run_{}/gamma.csv')

    
#     naive_result_uniform       =  get_result(get_prob_file_path(directory, delta, 'DNPL/scheduler_False/optimizer_sgd/lr_0.01/'))

#     proden_result_wo_uniform =  get_result(get_prob_file_path(directory, delta, 'PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'))
#     proden_result_w_uniform  =  get_result(get_prob_file_path(directory, delta, 'PRODEN/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'))

#     cavl_result_wo_uniform   =  get_result(get_prob_file_path(directory, delta, 'CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/'))
#     cavl_result_w_uniform    =  get_result(get_prob_file_path(directory, delta, 'CAVL/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/'))

#     lw_result_wo_uniform     =  get_result(get_prob_file_path(directory, delta, 'LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/cross_entropy/beta_2.0/'))
#     lw_result_w_uniform      =  get_result(get_prob_file_path(directory, delta, 'LW/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/cross_entropy/beta_2.0/'))

#     # cr_result_wo     =  get_result(get_prob_file_path(directory, delta, 'ICML22/' + 'supervised_loss_only/'))
#     cr_result_wo_uniform     =  get_result(get_prob_file_path(directory, delta, 'CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/weight_1.0_1.0_1.0/'))
#     cr_result_w_uniform      =  get_result(get_prob_file_path(directory, delta, 'CR/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/weight_1.0_1.0_1.0/'))

#     wo_proto_wo_contra_uniform   = get_result(get_prob_file_path(directory, delta, 'PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.0/'))
#     w_proto_wo_contra_uniform    = get_result(get_prob_file_path(directory, delta, 'PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.0/'))
#     wo_proto_w_contra_uniform    = get_result(get_prob_file_path(directory, delta, 'PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_False/contrast_weight_0.5/'))
#     w_proto_w_contra_uniform     = get_result(get_prob_file_path(directory, delta, 'PiCO/scheduler_True/optimizer_sgd/lr_0.01/confidence_True/contrast_weight_0.5/'))



#     result_mean_real = np.vstack((naive_result_real[0], proden_result_wo_real[0], proden_result_w_real[0], cavl_result_wo_real[0], cavl_result_w_real[0], lw_result_wo_real[0], lw_result_w_real[0], cr_result_wo_real[0], cr_result_w_real[0], wo_proto_w_contra_real[0], w_proto_w_contra_real[0]))
    
#     result_mean_uniform = np.vstack((naive_result_uniform[0], proden_result_wo_uniform[0], proden_result_w_uniform[0], cavl_result_wo_uniform[0], cavl_result_w_uniform[0], lw_result_wo_uniform[0], lw_result_w_uniform[0], cr_result_wo_uniform[0], cr_result_w_uniform[0], wo_proto_w_contra_uniform[0], w_proto_w_contra_uniform[0]))


#     # print(result_mean)
#     # exit(0)

#     keys = ['DNPL', 'PRODEN', 'PRODEN', 'CAVL', 'CAVL', 'LW', 'LW', 'CR', 'CR', 'PiCO', 'PiCO']

#     index = [i for i in range(len(keys))]

#     values_real     = [result_mean_real[i,0]    for i in range(len(result_mean_real))]
#     values_uniform  = [result_mean_uniform[i,0] for i in range(len(result_mean_uniform))]


#     plt.rcParams['axes.xmargin'] = 0.02
#     # fig, ax = plt.subplots()
#     fig= plt.figure()

#     result_wo_ld_real   = [result_mean_real[0,0]] + [result_mean_real[i,0] for i in range(1,len(result_mean_real)) if i%2==1]
#     result_w_ld_real    = [None] + [result_mean_real[i,0] for i in range(1,len(result_mean_real)) if i%2==0]

#     result_wo_ld_uniform   = [result_mean_uniform[0,0]] + [result_mean_uniform[i,0] for i in range(1,len(result_mean_uniform)) if i%2==1]
#     result_w_ld_uniform    = [None] + [result_mean_uniform[i,0] for i in range(1,len(result_mean_uniform)) if i%2==0]


#     xs = np.arange(6)
#     series1_real = np.array(result_wo_ld_real).astype(np.double)
#     s1mask_real = np.isfinite(series1_real)
#     series2_real = np.array(result_w_ld_real).astype(np.double)
#     s2mask_real = np.isfinite(series2_real)

    
#     series1_uniform = np.array(result_wo_ld_uniform).astype(np.double)
#     s1mask_uniform = np.isfinite(series1_uniform)
#     series2_uniform = np.array(result_w_ld_uniform).astype(np.double)
#     s2mask_uniform = np.isfinite(series2_uniform)




#     plt.plot(xs[s1mask_real], series1_real[s1mask_real], linestyle='-', marker='o', markersize=8, label ='real_world' + 'w/o LD')
#     plt.plot(xs[s2mask_real], series2_real[s2mask_real], linestyle='-', marker='o', markersize=8, label ='real_world' + 'w/'+'   '+'LD')

#     plt.plot(xs[s1mask_uniform], series1_uniform[s1mask_uniform], linestyle='-', marker='o', markersize=8, label ='classical' + 'w/o LD')
#     plt.plot(xs[s2mask_uniform], series2_uniform[s2mask_uniform], linestyle='-', marker='o', markersize=8, label ='classical' + 'w/'+'   '+'LD')


#     # bar_width = 0.3

#     # Create the bar chart
#     # plt.bar(xs[s1mask] - bar_width/2, series1[s1mask], width=bar_width, label ='w/o LD')
#     # plt.bar(xs[s2mask] + bar_width/2, series2[s2mask], width=bar_width, label ='w/'+'   '+'LD')

#     # Find min and max values from both series for y-axis scaling
#     min_value_real = min(np.min(series1_real[s1mask_real]), np.min(series2_real[s2mask_real]))
#     max_value_real = max(np.max(series1_real[s1mask_real]), np.max(series2_real[s2mask_real]))

#     min_value_uniform = min(np.min(series1_uniform[s1mask_uniform]), np.min(series2_uniform[s2mask_uniform]))
#     max_value_uniform = max(np.max(series1_uniform[s1mask_uniform]), np.max(series2_uniform[s2mask_uniform]))


#     # Adjust the y-axis limits (min_value - 0.05 to max_value + 0.05)
#     # plt.ylim(min_value - 0.01, max_value + 0.01)

#     plt.ylabel('Acc.', fontsize=12)
#     plt.xticks([0,1,2,3,4,5], ['DNPL' , 'PRODEN', 'CAVL', 'LW','CR', 'PiCO'])
#     plt.grid(axis='y')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()



























#
