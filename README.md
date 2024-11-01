# Loose Inertial Poser: Motion Capture with IMU-attached Loose-Wear Jacket
![](https://github.com/ZuoCX1996/Loose-Inertial-Poser/blob/main/figs/teaser.png)

Implementation of our CVPR 2024 paper "Loose Inertial Poser: Motion Capture with IMU-attached Loose-Wear Jacket". Including network weights, training and evaluation scripts.

Run **train_SemoAE.py** to train SeMo-AE.

Run **train_poser.py** to train pose estimation network.

Run **evaluation.py** to get ang Err and pos Err.

The real-world dataset is in the folder named **LIP_Dataset**, please notice that the original data frame rate is **60Hz**, and it was downsampled to 30Hz in our implementation.

The original data frame rate is 60Hz, and it was down sampled to 30Hz which is a 50% reduction.

The synthesized IMU data using TailorNet is available on Baidu Cloud:
https://pan.baidu.com/s/1UmFCHvt3pqIYixuuCqBqWg?pwd=nmyr