[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# KF-ALARM: Neighbor-Corroborated Kalman Filtering for Robust Distributed Multi-Object Tracking

This repository provides an implementation of the **KF-ALARM** framework, a Kalman-based extension of the **Average Likelihood for Attack-Resilient Multi-Object (ALARM)** principle for robust distributed multi-object tracking under adversarial measurement attacks.

The proposed method is described in the following preprint:  
🔗 https://doi.org/10.22541/au.177490735.56661375/v1

---

## 📄 Ghost Attack Modeling and Further Analysis

👉 `docs/Ghost Attack Modeling and Further Analysis.pdf`

This document presents the ghost attack simulation algorithm and complements the results reported in the paper, including:
- Simulation setup for ghost attacks
- Adaptive adversarial attack experiments  
- Benign-case performance analysis  
- Detailed per-iteration OSPA, and cardinality results  
- Additional visualizations and analyses  

All results are generated using the same experimental setup described in the manuscript.

---



## Related Work

Original ALARM repository:  
🔗 https://github.com/ghorbani-vahid/alarm-filtering

The KF-ALARM filter builds upon the Tracking Toolbox developed by Ba-Tuong Vo:  
🔗 https://ba-tuong.vo-au.com/codes.html

---

## Usage

1. Open MATLAB and navigate to the project root.  
2. Add required directories to the MATLAB path:
   ```matlab
   addpath('_common');
   addpath('_network');
