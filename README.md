# Partial Label Learning for Emotion Recognition from EEG
This is the implementation of [Partial Label Learning for Emotion Recognition from EEG](https://arxiv.org/pdf/2302.13170.pdf) in PyTorch (Version 1.11.0).
<p align="center">
  <img 
    width="600"
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
    height="210"
    src="/screenshot.png"
  >
</p>
In the bash script, decide whether to use the 'wait' command after each python execution command based on your available CPU and GPU resources. If you have multiple GPUs and many CPU cores, you can omit 'wait' and assign a GPU device ID to the environment variable 'CUDA_VISIBLE_DEVICES' for more efficient computation. If you have limited computational resources, leave the bash file as is.


 ## Document Description
 
- `\parsing`: includes argument specifications for all partial label learning methods. 
- `\backbone_model`: includes the backbone model for all the methods. 
- `\train_func`: includes the training steps for all partial label learning methods, including [DNPL](./train_func.py#L23-L44), [PRODEN](./train_func.py#L51-L82), [CAVL](./train_func.py#L89-L119), [LW](./train_func.py#L126-L233), [CR](./train_func.py#L240-L294), and [PiCO](./train_func.py#L301-L336), for EEG representation learning in emotion recognition tasks.
- `\main`: includes implementation of experiments for several state-of-the-art partial label learning methods: [DNPL](https://ieeexplore.ieee.org/document/9414927), [PRODEN](https://dl.acm.org/doi/10.5555/3524938.3525541), [CAVL](https://openreview.net/forum?id=qqdXHUGec9h), [LW](http://proceedings.mlr.press/v139/wen21a.html), [CR](https://proceedings.mlr.press/v162/wu22l.html), and [PiCO](https://openreview.net/forum?id=EhYjZy6e1gJ). In the paper, we conducted experiments with and without label disambiguation for most of these methods. To make the code easier to understand, the label disambiguation process is placed in the 'confidence_update' function within the class of each method. To enable label disambiguation in an experiment, simply add the '--use-confidence' flag to the command line. The DNPL method is the only exception and does not use label disambiguation.  
 - `\polar_coordinate`: inlcudes generation of simlarities among five emotions based on the emotion wheel.
 - `\utils`: includes model paramter initialization, methods for generating candidate labels based on classical (uniform distribution) and real-world (emotion similarity) settings, a custom PyTorch data loader, as well as contrastive learning loss functions. 

 
 
If you find this material useful, please cite the following article:

## Citation

```
@article{zhang2023partial,
  title={Partial Label Learning for Emotion Recognition from EEG},
  author={Zhang, Guangyi and Etemad, Ali},
  journal={arXiv preprint arXiv:2302.13170},
  year={2023}
}

```

## Contact
The sources of randomness are controlled, ensuring that all results presented in the paper can be replicated using the codes provided in this repository. Should you have any questions, please feel free to contact me at [guangyi.zhang@queensu.ca](mailto:guangyi.zhang@queensu.ca).



<!-- <img src="/doc/architecture.pdf" width="300" height="200">
 -->
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fguangyizhangbci%2FPLL-Emotion-EEG&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>

