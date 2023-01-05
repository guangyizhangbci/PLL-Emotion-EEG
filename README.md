# Partial Label Learning for Emotion Recognition from EEG
This is the implementation of [Partial Label Learning for Emotion Recognition from EEG](arxiv address) in PyTorch (Version 1.11.0).
<p align="center">
  <img 
    width="900"
    height="310"
    src="/framework.jpg"
  >
</p>



This repository contains the source code of our paper, using a large publicly avaiable dataset, namely [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html): 16 participants were involved in experiments with videos as emotion stimuli (five emotions: happy/sad/disgust/neutral/fear). 62 EEG recordings were collected with a sampling frequency of 1000Hz.


## Prerequisites
Please follow the steps below in order to be able to train our models:


1 - Install Requirements

```
pip3 install -r ./requirements.txt
```

2 - Download the extracted differential entropy features and labels provided by the dataset. Follow the [data loader](./load_data.py) code while strictly following the official dataset description.




4 - Usage:


5. bash file

 ## Document Description
 
- `\parsing`: argparse module including argument specifications for all partial label learning methods. 
- `\backbone_model`: backbone model for all the methods, including PiCO which requires both encoder and momentum encoder.  
- `\train_func`:  training steps for all partial label learning methods, namely [DNPL](./train_func.py#L23-L44), [PRODEN](./train_func.py#L51-L82), [CAVL](./train_func.py#L89-L119), [LW](./train_func.py#L126-L233), [CR](./train_func.py#L240-L294), and [PiCO](./train_func.py#L301-L336) for EEG representation learning in emotion recognition tasks.
- `\main`: implementation of experiment set-up for all recent state-of-the-art partial label learning methods, [DNPL](https://ieeexplore.ieee.org/document/9414927), [PRODEN](https://dl.acm.org/doi/10.5555/3524938.3525541), [CAVL](https://openreview.net/forum?id=qqdXHUGec9h), [LW](http://proceedings.mlr.press/v139/wen21a.html), [CR](https://proceedings.mlr.press/v162/wu22l.html), and [PiCO](https://openreview.net/forum?id=EhYjZy6e1gJ). In the paper, we investigated majority of the methods with and without label disambiguation. To make code easier for reader to understand, we place the label disambiguation process in the 'confidence_update' function in the class of each method. Therefore, we can run the experiments by simply adding '--use-confidence' in the command line for enabling the label disambiguation steo. Method DNPL is the only exception which does not employ label disambiguation.  
 - `\polar_coordinate`: generation of simlarities among five emotions based on the emotion wheel.
 - `\utils`: model paramters initialization, candidate label generation methods based on classical (uniform distribution) and real-world (emotion similarity) settings, customized PyTorch data loader, as well as contrastive learning loss functions. 
 
 
 
 
