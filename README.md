# Partial Label Learning for Emotion Recognition from EEG
This is the implementation of [Partial Label Learning for Emotion Recognition from EEG](arxiv address) in PyTorch (Version 1.11.0).
<p align="center">
  <img 
    width="900"
    height="310"
    src="/framework.jpg"
  >
</p>



This repository contains the source code of our paper, using a large publicly avaiable dataset, [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html). 16 participants were involved in experiments with videos as emotion stimuli (five emotions: happy/sad/disgust/neutral/fear). 62 EEG recordings were collected with a sampling frequency of 1000Hz.


## Prerequisites
Please follow the steps below in order to be able to train our models:


1 - Install Requirements

```
pip3 install -r ./requirements.txt
```

2 - Download the extracted differential entropy features and labels provided by the dataset. Follow the [data loader](./load_data.py) code while strictly following the official dataset description.


3 - Usage: In order to have a user-friendly interface, we provide bash file for runing all the partial label learning experiments under various settings.  Run the command to excute the bash file:
```
bash ./run.sh
```
Enter the values specified in the terminal hint. These may include integers, strings, or boolean values. For the method name, choose one of 'DNPL', 'PRODEN', 'CAVL', 'CR', 'LW', or 'PiCO'. For label disambiguation, enter 'true' or 'false'. For partial label type, enter 'uniform' or 'emotion'. For loss name, enter 'sigmoid' or 'cross_entropy'. For beta values in the LW method, enter '0.0', '1.0', or '2.0'. For the PiCO method, enter 'true' or 'false' to determine whether contrastive learning should be used. An example is shown in the screenshot.

<p align="center">
  <img 
    width="900"
    height="310"
    src="/screenshot.png"
  >
</p>
In the bash fash, please choose whether to use the 'wait' command after each python exexcution command according to the CPU and GPU resource you have. If you have mutiple GPU and many CPU cores, you can choose to remove 'wait' and assign GPU device id to CUDA_VISIBLE_DEVICES=id for efficient computation. If your computation source is limited, keep the bash file as it is.


 ## Document Description
 
- `\parsing`: includes argparse module containing argument specifications for all partial label learning methods. 
- `\backbone_model`: includes the backbone model for all the methods. 
- `\train_func`: includes training steps for all partial label learning methods, namely [DNPL](./train_func.py#L23-L44), [PRODEN](./train_func.py#L51-L82), [CAVL](./train_func.py#L89-L119), [LW](./train_func.py#L126-L233), [CR](./train_func.py#L240-L294), and [PiCO](./train_func.py#L301-L336) for EEG representation learning in emotion recognition tasks.
- `\main`: includes implementation of experiment set-up for all recent state-of-the-art partial label learning methods, [DNPL](https://ieeexplore.ieee.org/document/9414927), [PRODEN](https://dl.acm.org/doi/10.5555/3524938.3525541), [CAVL](https://openreview.net/forum?id=qqdXHUGec9h), [LW](http://proceedings.mlr.press/v139/wen21a.html), [CR](https://proceedings.mlr.press/v162/wu22l.html), and [PiCO](https://openreview.net/forum?id=EhYjZy6e1gJ). In the paper, we investigated majority of the methods with and without label disambiguation. To make code easier for reader to understand, we place the label disambiguation process in the 'confidence_update' function in the class of each method. Therefore, we can run the experiments by simply adding '--use-confidence' in the command line for enabling the label disambiguation process. Method DNPL is the only exception which does not employ label disambiguation.  
 - `\polar_coordinate`: inlcudes generation of simlarities among five emotions based on the emotion wheel.
 - `\utils`: includes model paramters initialization, candidate label generation methods based on classical (uniform distribution) and real-world (emotion similarity) settings, customized PyTorch data loader, as well as contrastive learning loss functions. 

 
 
