# IJCAI_2021_Continual_Crowd_Counting_Challenge

## Baseline

This repo helps the challengers to access dataset and baseline conveniently. The workshop/challenge website is located at:\
https://sites.google.com/view/sscl-workshop-ijcai-2021/

The baseline is inspired from the unofficial implementation of CVPR 2016 paper "Single Image Crowd Counting via Multi Column Convolutional Neural Network" at https://github.com/svishwa/crowdcount-mcnn. However, the repo is updated with latest pytorch version and unsupervised training part.

## Dataset Description

The dataset is designed to test continual semi-supervised learning for crowd counting in video frames. This benchmark consists of the union of three existing datasets (Mall, UCSD and FDST), augmented with the relevant ground truth in terms of density maps.

The dataset can be downloaded from link below:\
https://drive.google.com/file/d/1phQi86FvLXBoOeh9jeZ_-MomUbcGW04K/view?usp=sharing

The train and validation (without groundtruth) is arranged as follows:\
- Every dataset folder (say fdst) consists of two sub folders (train, val).
- A subfolder (say train) may contain input and gt as subfolder.
- Validation fold does not have a gt folder.
                      
Change the following paths in the train.py to train on respective datasets:\
\
train_path = './data/fdst/train/input/' \
train_gt_path = './data/fdst/train/gt/' \
val_path = './data/fdst/val/input/' 
                      
## Running the baseline
\
The training can be started by running:\
                                      python train.py 
