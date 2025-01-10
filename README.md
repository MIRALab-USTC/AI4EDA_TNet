# Towards Next-Generation Logic Synthesis: A Scalable Neural Circuit Generation Framework

This is the code for our paper "Towards Next-Generation Logic Synthesis: A Scalable Neural Circuit Generation Framework". Zhihai Wang, Jie Wang*, Qingyue Yang, Yinqi Bai, Xing Li, Lei Chen, Jianye Hao,Mingxuan Yuan, Bin Li, Yongdong Zhang, Feng Wu. NeurIPS 2024. [Paper](https://openreview.net/pdf?id=ZYNYhh3ocW)

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

@inproceedings{

wang2024towards,

title={Towards Next-Generation Logic Synthesis: A Scalable Neural Circuit Generation Framework},

author={Zhihai Wang and Jie Wang and Qingyue Yang and Yinqi Bai and Xing Li and Lei Chen and Jianye HAO and Mingxuan Yuan and Bin Li and Yongdong Zhang and Feng Wu},

booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},

year={2024},

url={https://openreview.net/forum?id=ZYNYhh3ocW}

}
