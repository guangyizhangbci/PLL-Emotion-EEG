import numpy as np
import pickle
import os


load_path = '/media/patrick/DATA/SEED_V/EEG_DE_features/'
save_path = '/media/patrick/DATA/SEED_V/EEG/'


data_npz = np.load(os.path.join(load_path, '1_123.npz'))
# data = pickle.loads(data_npz['data'])
# data = list(data.values())

data = pickle.loads(data_npz['label'])
data = list(data.values())

# an_array = np.array(data)
# print(an_array[0])
# exit(0)
for subject_num in range(1, 17):
    for fold_num in range(1, 4):

        data_npz = np.load(os.path.join(load_path, '{}_123.npz').format(subject_num))

        data  = pickle.loads(data_npz['data'])
        label = pickle.loads(data_npz['label'])

        data  = np.array(list(data.values()))
        label = np.array(list(label.values()))


        temp_data  = np.zeros((0, 310))
        temp_label = np.zeros((0, 1))

        for session_num in range(0, 3):
            start_trial_num, end_trial_num = 5*(fold_num-1), 5*fold_num

            start_trial_num = start_trial_num + 15*session_num
            end_trial_num   = end_trial_num + 15*session_num


            for trials_num in range(start_trial_num, end_trial_num):
                print(session_num, trials_num)
                temp_data  = np.vstack((temp_data,   data[trials_num]))
                temp_label = np.vstack((temp_label,  np.expand_dims(label[trials_num], 1)))


        np.save(os.path.join(save_path, 'de_{}_{}').format(subject_num, fold_num), temp_data)
        np.save(os.path.join(save_path, 'label_{}_{}').format(subject_num, fold_num), temp_label)


























#
