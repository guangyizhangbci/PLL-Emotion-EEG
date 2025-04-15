"""
Created on Thu May 14 15:32:30 2022

@author: patrick
"""
from __future__ import print_function
import numpy as np
import math
import random
from math import log, e
import matplotlib.pyplot as plt

import pickle

# data_addr  = '/home/patrick/SEED_V/EEG/de_{}_{}.npy'      # subject_No, Fold_No
# label_addr = '/home/patrick/SEED_V/EEG/label_{}_{}.npy'   # subject_No, Fold_No


# X = np.load(data_addr.format(1, 1))
# Y = np.load(label_addr.format(1, 1))

# print(Y)
data_addr   = '/home/patrick/Documents/SEED_V_code/1_123.npz'      # subject_No, Fold_No

data_npz = np.load(data_addr)
print(data_npz.files)

data = pickle.loads(data_npz['data'])
label = pickle.loads(data_npz['label'])

# Define the label dictionary
label_dict = {0: 'Disgust', 1: 'Fear', 2: 'Sad', 3: 'Neutral', 4: 'Happy'}


# Define an empty list to collect data for Session 1 only
session_1_data   = []
session_1_labels = []

for i in range(45):
    # Access data and label for each trial
    trial_data = data[i]  # Assuming each `trial_data` is already in shape [62, 5]
    trial_label = label[i][0]  # Assuming `label[i][0]` gives the label index
    
    # Determine session and trial numbers
    session_num = i // 15 + 1
    trial_num = i % 15 + 1

    # Check if the trial is part of Session 1
    if session_num == 1:
        print('Session {} -- Trial {} -- EmotionLabel: {}'.format(session_num, trial_num, label_dict[trial_label]))
        print('Corresponding Data Shape:', trial_data.shape)
        
        # Append the data for Session 1 to the list without reshaping (assuming [62, 5] shape)
        session_1_data.append(trial_data)
        session_1_labels.append(trial_label)

# Stack all session 1 data into a single 3D array (15 trials, 62 channels, 5 frequency bands)
session_1_data = np.concatenate(session_1_data, axis=0) 
session_1_labels = np.array(session_1_labels)  # Convert labels to a NumPy array

print("Session 1 Data Shape:", session_1_data.shape)    # Expected shape: (15, 62, 5) or (n, 62, 5)
print("Session 1 Labels Shape:", session_1_labels.shape)  # Expected shape: (15,)
print("Session 1 Labels:", session_1_labels)  # Display labels for Session 1


negative_values = session_1_data[session_1_data < 0]
print("Negative values in the matrix:", negative_values, len(negative_values))


trial_ranges = {
            'Happy': (0, 18),
            'Fear': (18, 42),
            'Neutral': (42, 101),
            'Sad': (101, 147),
            'Disgust': (147, 183),
            'Happy_2': (183, 247),
            'Fear_2': (247, 321),
            'Neutral_2': (321, 338),
            'Sad_2': (338, 404),
            'Disgust_2': (404, 439),
            'Happy_3': (439, 482),
            'Fear_3': (482, 525),
            'Neutral_3': (525, 583),
            'Sad_3': (583, 643),
            'Disgust_3': (643, 681)
        }

# Define emotion labels for each base emotion type
emotion_labels = {
    'Disgust': '0',
    'Fear': '1',
    'Sad': '2',
    'Neutral': '3',
    'Happy': '4'
}

data_reshaped = session_1_data.reshape(session_1_data.shape[0], 62, 5)
new_data = data_reshaped.transpose(0, 2, 1)
new_data_310 =new_data.reshape(session_1_data.shape[0], 310)

# new_data = session_1_data.reshape(new_data.shape[0], 5, 62)
# new_data_310 =session_1_data
# Frequency band labels


# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the heatmap
cax = ax.imshow(new_data_310.T, aspect='auto', cmap='jet', origin='lower')
fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.02)

# Add frequency band labels on the y-axis


# Set primary y-axis ticks for frequency bands (centered within 310 channels)

channel_ticks = np.linspace(0, 300, 7)  # Ticks at 0, 50, 100, ..., 300
ax.set_yticks(channel_ticks)
ax.set_yticklabels(channel_ticks.astype(int))


# Secondary y-axis for evenly spaced channel ticks on the left
ax2 = ax.twinx()
ax2.spines['left'].set_position(('outward', 30))  # Move secondary y-axis to the left
ax2.yaxis.set_label_position('left')
ax2.spines['left'].set_visible(False)

ax2.yaxis.tick_left()
# ax2.tick_params(left=False, labelleft=False)  # Hide ticks and labels
# Set y-ticks for the secondary y-axis

bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
channel_count = 62  # Each band has 62 channels
yticks = [(i * channel_count + channel_count // 2) for i in range(len(bands))]  # Midpoints of each band
print(yticks)
ax2.set_yticks(yticks)
ax2.set_yticklabels(bands, rotation=90, verticalalignment='center')
ax2.set_ylim(0, 310)

ax2.set_ylabel("Frequency Bands",  labelpad=10)


time_ticks = np.linspace(0, 2500//4 , num=6)  # Set 10 ticks as an example


time_labels = (time_ticks * 4).astype(int)  # Scale by 4 to show time in seconds

ax.set_xticks(time_ticks)
ax.set_xticklabels(time_labels)
ax.set_xlabel("Time (seconds)")

# Create a secondary x-axis at the top for the emotion labels
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())  # Align the top axis with the main axis



# Add boundaries and labels at the top x-axis for each emotion
for trial, (start, end) in trial_ranges.items():
    # Strip off any suffix (_2, _3) to get the base emotion label
    base_trial = trial.split('_')[0]  # 'Disgust_2' becomes 'Disgust'
    label = emotion_labels.get(base_trial, '')

    # Mark the boundary with a vertical line
    ax_top.axvline(x=start-0.5, color='black', linestyle='--', linewidth=1.0,  ymin=1.0, ymax=1.03, clip_on=False) 

    # Ensure we are using ax_top.transAxes for positioning relative to the top axis
    midpoint = (start + end) / 2
    ax_top.text(midpoint, 311, label, ha='center', va='bottom', fontsize=10, color='black')


ax_top.axvline(x=session_1_data.shape[0]-0.5, color='black', linestyle='--', linewidth=1.0,  ymin=1.0, ymax=1.03, clip_on=False) 

# Hide ticks on the top axis and add a label for clarity
ax_top.set_xticks([])  # Remove tick marks on the top x-axis
ax_top.set_xlabel("Emotion Codes ('0': Disgust, '1': Fear, '2': Sad, '3': Neutral, '4': Happy)", labelpad=20)


# plt.tight_layout()
# plt.show()
# exit(0)


import mne
from mne.channels import make_standard_montage
from mne.viz import plot_topomap

montage = make_standard_montage('standard_1020')
custom_order = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4",
    "PO6", "PO8", "PO9", "O1", "Oz", "O2", "PO10"
]
# CB1 --> PO9,  CB2--> PO10, adjustment accoding to standard 1020 and MNE 1020



# Create the info object with your custom order
info = mne.create_info(ch_names=custom_order, sfreq=200, ch_types="eeg")
info.set_montage(montage)
# info.set_montage(montage, on_missing='ignore')
# Get channel positions in your custom order
pos = np.array([info['chs'][info.ch_names.index(ch)]['loc'][:2] for ch in custom_order])



# Create a 1x5 subplot layout
fig, axes = plt.subplots(5, 5, figsize=(20, 20))  # Adjust figsize as needed


happy_start_index = 10 # NEED to be in the range of [0, 17]

emotion_index = [happy_start_index,        happy_start_index+18,    happy_start_index+18+24, happy_start_index+18+24+59, happy_start_index+18+24+59+46]
emotion_name  = ['Happy',                  'Fear',                  'Neutral',               'Sad',                      'Disgust']



vmin = np.min(new_data)
vmax = np.max(new_data)

# Loop over each frequency band to plot on each axis
for i in range(5):
    for freq_band in range(5):
        # Plot the topomap for each frequency band on the respective subplot axis
        im, _ = plot_topomap(new_data[emotion_index[i], freq_band, :], pos, 
                            cmap='viridis', vlim=(vmin, vmax), sphere=(0., 0., 0., 0.12), axes=axes[i, freq_band], show=False) # names=custom_order,

    # Add a colorbar for each subplot

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.05)

# Add labels
for i, emotion in enumerate(emotion_name):
    axes[i, 0].set_ylabel(emotion, rotation=90, ha='center', va='center', fontsize=12, labelpad=15)
    
for j, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
    axes[0, j].set_title(band)


# Display all topomaps
# plt.tight_layout()
plt.show()


# # Loop over emotions (rows) and bands (columns)
# for i in range(5):  # Emotions
#     for freq_band in range(5):  # Frequency bands
#         # Get data: [emotion, band, channels]
#         plot_data = new_data[emotion_index[i], freq_band, :]
        
#         im = plot_topomap(
#             plot_data,
#             pos,
#             cmap='viridis',
#             vlim=(vmin, vmax),  # Enforce global scaling
#             sphere=(0., 0., 0., 0.095),  # MNE standard
#             axes=axes[i, freq_band],
#             show=False
#         )[0]

# # Add SINGLE colorbar for all subplots
# cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
# cbar.set_label('Power (dB)')

