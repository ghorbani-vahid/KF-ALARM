[![DOI](https://zenodo.org/badge/1177465975.svg)](https://doi.org/10.5281/zenodo.18932375)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# KF-ALARM: Neighbor-Corroborated Kalman Filtering for Robust Distributed Multi-Object Tracking

This repository contains an implementation of the **KF-ALARM** framework, a Kalman-based extension of the **Average Likelihood for Attack-Resilient Multi-Object (ALARM)** principle for robust distributed multi-object tracking under adversarial measurement attacks.


Original ALARM repository: [https://github.com/ghorbani-vahid/alarm-filtering]

The KF-ALARM filter is built on top of the Tracking Toolbox provided by Ba-Tuong Vo:  
https://ba-tuong.vo-au.com/codes.html


## Usage

1. **Open MATLAB** and change directory to the project root.  
2. **Ensure required directories are on the MATLAB path**:
   ```matlab
   addpath('_common');
   addpath('_network');
3. **Run the main.m script**:

