# EGMN

================================================


Overview
--------

PyTorch implementation of our paper "Multi-Granularity Distribution Modeling for Video Watch Time Prediction via Exponential-Gaussian Mixture Network".

This repository also contains the implementation of all baselines and the processing of KuaiRec dataset mentioned in our paper.

_**Other datatsets** and **more in-depth details** mentoined in our experiment section will be synchronized in this repository if the manuscript is accepted._

Dependencies
------------

Install Pytorch 2.1.0, using pip or conda, should resolve all dependencies (pandas, numpy, sklearn).

Tested with Python 3.10.12, but should work with 3.x as well.

Tested on CPU or GPU.

Public Dataset
-------

You can download the public dataset from following links:
* [KuaiRec](https://kuairec.com/)


Raw data need to be preprocessed before using. The data preprocessing scripts are given in `dataset/kuairec/kuairec_process.py`.

How to Use
----------
`model/*`: Implementation of various models.

`run_*.py`: The starting point for running each method.

You can run the program with these command examples:
<br>
<br>
`python run_cread.py --dataset_name kuairec` : evaluate **CREAD** on  KuaiRec dataset
<br>
<br>
`python run_egmn.py --dataset_name kuairec`: evaluate proposed **EGMN** on the KuaiRec dataset
<br>
<br>
The program will print the MAE, XAUC and KL-Divergend of evaluation. 
Some other settable parameters could be found in the `./run_*.py` file.
