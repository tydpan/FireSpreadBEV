# Investigate Fire Spread by Bird-Eye View
by **Tai-Yu Pan**

Note: The architecture is forked from the officail implementation of [U-Net](https://github.com/milesial/Pytorch-UNet). The code for this project can be found in VLS.py focal_loss.py and train_contest.py, which inlcludes data loader for VLS dataset, data augmentations, evaluation metrics, and the training pipleline. 

## Introduction
The emergent behavior of wildfires can be complex because it’s a result of a set of non-liner interactions, including combustion, atmospheric dynamics, and a multi-phase turbulent flow. Directly modeling it in the 3D volume is as a result a difficult task. However, given the fact that in most of times we only care about the spread on the ground, modeling the behavior of wildfires from bird-eye view is more feasible. The goal is to answer the question: which factors are important to the spread of wildfires.

## Usage
```
python train_contest.py -e 200 -b 10 -l 1e-4 -c 6 --bilinear
```