# Loose Inertial Poser: Motion Capture with IMU-attached Loose-Wear Jacket
![](https://github.com/ZuoCX1996/Loose-Inertial-Poser/blob/main/figs/teaser.png)

Implementation of our CVPR 2024 paper "Loose Inertial Poser: Motion Capture with IMU-attached Loose-Wear Jacket". Including network weights, training and evaluation scripts.

Run [train_SemoAE.py](./train_SemoAE.py) to train SeMo-AE.

Run [train_poser.py](./train_poser.py) to train pose estimation network.

Run [evaluation.py](./evaluation.py) to get ang Err and pos Err.

Run [vis_pose.py](./vis_pose.py) to visualize pose predictions or ground truth.

## Live Demo 
1. Prepare the pose visualization unity project. You can find it at TransPose project page: https://github.com/Xinyu-Yi/TransPose
2. Run [live demo.py](./live%20demo.py)
3. Run unity visualization

## LIP Dataset
The real-world dataset is in the folder named **LIP_Dataset**, please notice that the original data frame rate is **60Hz**, and it was downsampled to 30Hz in our implementation.

The original data frame rate is 60Hz, and it was down sampled to 30Hz which is a 50% reduction.

## Synthesized IMU data
The synthesized IMU data using TailorNet is available at the following links:

Baidu Cloud: https://pan.baidu.com/s/1UmFCHvt3pqIYixuuCqBqWg?pwd=nmyr

OneDrive: https://1drv.ms/f/c/d6cd78cd7ce83043/Etl3HdtmHl5Dn05NQ-ZKScMBYsCo4gTef7Io3JVZb07nxQ?e=UMXx0e

The synthesized IMU data ON AMASS (Match the TailorNet syn IMU data) is available at the following links:
:
https://1drv.ms/f/c/d6cd78cd7ce83043/Eqz0kup3E0NEpr0j0A9R8DIBb4Jr0D-PS2MnQtYKJjCdzQ?e=urBJ6m
