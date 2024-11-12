# Towards Next-Generation Logic Synthesis: A Scalable Neural Circuit Generation Framework

This is the code for our paper "Towards Next-Generation Logic Synthesis: A Scalable Neural Circuit Generation Framework". Zhihai Wang, Jie Wang, Qingyue Yang, Yinqi Bai, Xing Li, Lei Chen, Jianye Hao,Mingxuan Yuan, Bin Li, Yongdong Zhang, Feng Wu. NeurIPS 2024.

## Environment
- python 3.7
- pytorch
- numpy

Or alternatively, to build the environment from a file,

`bash environment.sh`

## Usage
Put the truth tables under the `truthtable/` directory. Add the net size in `experiments/net_config.py`

Here is an example for generate a circuit.

`bash run.sh`

## Citation
If you find this code useful, please consider citing the following papers.