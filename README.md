# Partial Label Learning for Emotion Recognition from EEG
This is the implementation of [Partial Label Learning for Emotion Recognition from EEG](arxiv address) in PyTorch (Version 1.11.0).


This repository contains the source code of our paper, using a large publicly avaiable dataset, namely [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html): 16 participants were involved in experiments with videos as emotion stimuli (five emotions: happy/sad/disgust/neutral/fear). 62 EEG recordings were collected with a sampling frequency of 1000Hz.


## Prerequisites
Please follow the steps below in order to be able to train our models:


1 - Install Requirements

```
pip3 install -r ./requirements.txt (need to add)
```

2 - Download the extracted differential entropy features and labels provided by the dataset. Follow the [data processing and feature extraction](./load_data.py) code while strictly following the official dataset description.


3 - Save the extracted features and labels accordingly (e.g., './EEG/de_1_1.npy' and './EEG/label_1_1.npy'). Move EEG features and corresponding labels to the address shown in [here](./main.py#L37-L38). 


4 - Usage:


5. bash file

 ## Document Description
