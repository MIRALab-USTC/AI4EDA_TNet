#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

mkdir -p log

num_iterations=5000
eval_freq=100
batch_size=10
save_acc_threshold=True
descent_tau=True
truthtable_file="example.txt"
log_tb=True

cur_date=$(date +%y-%m-%d,%H:%M:%S)

python -u experiments/main.py \
    --num_iterations ${num_iterations} \
    --eval_freq ${eval_freq} \
    --truthtable_file ${truthtable_file} \
    --log_tb ${log_tb} \
    --batch_size ${batch_size} \
    --save_acc_threshold ${save_acc_threshold} \
    --descent_tau ${descent_tau} > ./log/"${cur_date}"_example.log 2>&1 &