# CIS-few-shot

A project to validate ClipAdapter's performance using the open-source CIS dataset and a custom worker dataset.

## Overview

This project leverages the open-source CIS dataset and a custom worker dataset to evaluate the performance of **ClipAdapter**. You can view the results and experiment details in the `rebar.ipynb` notebook.

## Guide
### 1. Init
First, clone the main repository and the ClipAdapter repository:
```bash
git clone https://github.com/gaobiaoli/CIS-few-shot.git
cd CIS-few-shot
git clone https://github.com/gaobiaoli/ClipAdapter.git
```
### 2. Dataset Preparation
```bash
cd CIS-few-shot/dataset
tar -xvf rebar_tying.gz
```
### 3.Validation
Turn to jupter notebook: rebar.ipynb
