[![DOI](https://zenodo.org/badge/1177465975.svg)](https://doi.org/10.5281/zenodo.18932375)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# KF-ALARM: Neighbor-Corroborated Kalman Filtering for Robust Distributed Multi-Object Tracking

This repository provides an implementation of the **KF-ALARM** framework, a Kalman-based extension of the **Average Likelihood for Attack-Resilient Multi-Object (ALARM)** principle for robust distributed multi-object tracking under adversarial measurement attacks.

The proposed method is described in the following preprint:  
🔗 https://doi.org/10.22541/au.177490735.56661375/v1

---

## 📄 Extended Supplementary Material

An extended supplementary document accompanying the paper is provided:

👉 `docs/KF-ALARM_Extended_Supplementary_Material.pdf`

This document complements the results presented in the paper and includes:
- Adaptive adversarial attack experiments  
- Benign-case performance analysis  
- Detailed per-iteration OSPA, and cardinality results  
- Additional visualizations and analyses  

All results are generated using the same experimental setup described in the manuscript.

---

## 🔗 Relation to Paper

This repository complements the accompanying paper by providing implementation details and extended experimental results that support and expand the reported findings.

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
