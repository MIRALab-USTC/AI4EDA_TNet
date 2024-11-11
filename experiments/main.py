import argparse
import math
import random
import os
import json
import sys

import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

from truth_table_datasets import TruthTableDataset
from difflogic import TNet
from net_config import tnet_size


def read_file_info(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if len(lines) == 0:
                print("None file content")
                return None, None
            first_line = lines[0].strip()
            m = len(first_line)
            n = len(lines)
            if not all(bit in '0' or '1' for line in lines for bit in line):
                print("Error in file content, only 0 and 1 are allowed")
                return None, None
            if m > 0 and (m & (m - 1)) == 0:
                return n, int(m.bit_length() - 1)
            else:
                print("Error in file content, the number of input bits should be a power of 2")
                return None, None
    except Exception as e:
        print(f"Read truthtable error: {e}")
        return None, None

def load_dataset(args):
    output_bits, input_bits = read_file_info(args.dataset_dir)
    args.input_bits = input_bits
    args.output_bits = output_bits
    print(f'data: {args.dataset_dir} \n in bit = {input_bits}, output_bits bit = {output_bits}')
    train_set = TruthTableDataset(random_data=False, input_nums=input_bits, truth_table_file = args.dataset_dir, truth_flip = args.truth_flip)
    test_set = train_set

    if input_bits <= args.batch_size:
        bs = 2 ** input_bits
    else:
        bs = 2 ** args.batch_size
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, drop_last=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2 ** input_bits, shuffle=False, pin_memory=True, drop_last=False)

    return train_loader, test_loader, test_set

def get_model(args):

    model = TNet(in_dim = args.input_bits,
        out_dim = args.output_bits,
        up_k = args.up_k,   
        up_l = args.up_l,
        down_k = args.down_k,
        down_l = args.down_l, 
        tau = 1.0,
        descent_layer = args.regularized_skip_connection,
        descent_layer_in = args.regularized_skip_connection)

    print(model)
    model = model.to('cuda')
    return model

def cal_wrong_rate(model, train_loader):
    wrong_rate_sum = 0
    count = 0
    for _, (x_bs, y_bs) in enumerate(train_loader):
        x_bs = x_bs.to('cuda').to(torch.float32)
        y_bs = y_bs.to('cuda').to(torch.float32)
        model.train(False)
        x_out = model(x_bs)
        different_elements = (x_out != y_bs)
        wrong_items = torch.sum(different_elements)
        wrong_rate_sum += wrong_items / (y_bs.shape[0] * y_bs.shape[1])
        count += 1

    return wrong_rate_sum/count

def train(model, train_loader, loss_fn, optimizer, args, epoch, iteration=0, best_acc=0):

    loss_acc_each_y = torch.zeros([args.output_bits], device='cuda')
    loss_acc_sum = 0

    if args.boolean_hardness_aware_loss_2 and best_acc >= 0.99:
        wrong_rate = cal_wrong_rate(model, train_loader)

    for _, (x_bs, y_bs) in enumerate(train_loader):
        iteration += 1
        x_bs = x_bs.to('cuda').to(torch.float32)
        y_bs = y_bs.to('cuda').to(torch.float32)

        model.train()
        x = model(x_bs)
    
        loss_acc_each_node = loss_fn(x, y_bs.float())
        o_y = torch.abs(x.detach()-y_bs)

        if args.boolean_hardness_aware_loss_1 and best_acc > 0.95: 
            o_y = torch.abs(x.detach()-y_bs) # 256*5
            hard_example_weight = torch.exp(2 * (o_y - 0.3))


            loss_acc_each_node = hard_example_weight * loss_acc_each_node

        loss_acc_each_column = torch.sum(loss_acc_each_node, dim=0)
        loss_acc_each_y += loss_acc_each_column
        loss_acc = torch.sum(loss_acc_each_column)
        

        if args.boolean_hardness_aware_loss_2 and best_acc >=0.99:
            if wrong_rate > 0:
                loss_acc = loss_acc / (wrong_rate*args.boolean_hardness_aware_loss_beta)

        loss_acc_sum+=loss_acc

        optimizer.zero_grad()
        loss_acc.backward()
        for param in model.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)

        optimizer.step()

        if args.save_model and iteration % args.save_model_freq == 0:
            print(f"Save model")
            torch.save(model, os.path.join(args.save_dir, f"{iteration}.pth"))


    loss_acc = loss_acc_sum
    loss_acc_each_y = loss_acc_each_y
    
    return loss_acc.item()

def eval(model, test_loader, mode, args):
    with torch.no_grad():
        model.train(mode=mode)

        wrong_elements = 0
        total_elements = 0

        correct_rows = 0
        total_rows = 0
        wrong_column = torch.zeros([args.output_bits]).to('cuda')
        different_elements_list = torch.empty([0], device='cuda')

        for _, (x_bs, y_bs) in enumerate(test_loader):
            x_bs = x_bs.to('cuda').to(torch.float32)
            y_bs = y_bs.to('cuda').to(torch.float32)

            x_out = model(x_bs)

            different_elements = (x_out != y_bs)

            wrong_elements += torch.sum(different_elements).item()
            total_elements += y_bs.numel()

            # calculation of row accuracy
            row_errors = torch.sum(x_out != y_bs, dim=1)
            row_correct = (row_errors == 0).to(torch.float32)
            correct_rows += torch.sum(row_correct).item()
            total_rows += x_bs.shape[0]

        element_accuracy = 1 - wrong_elements / total_elements
        row_accuracy = correct_rows / total_rows
        print(f"all_wrongs = {wrong_elements}")
        print(f"element_accuracy={element_accuracy}, row_accuracy={row_accuracy}")
      
    return element_accuracy


def get_args():
    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('--truthtable_file', type=str)
    parser.add_argument('--seed', type=int, default=0) 
    parser.add_argument('--truth_flip', type=bool, default=True, help='flip the truth table from order "00 01 10 11" to "11 01 10 00", suit for "cec" in abc') #

    parser.add_argument('--batch_size', type=int, default=10, help='batch size (default: 2^10)')
    parser.add_argument('--learning_rate', type=float, default=0.02, help='learning rate') 
    parser.add_argument('--num_iterations', type=int, default=100000) 
    parser.add_argument('--eval_freq', type=int, default=100) 

    # parser.add_argument('--down_k', type=int) # 网络初始宽度，重要参数
    # parser.add_argument('--down_l', type=int) # 网络初始深度，重要参数
    # parser.add_argument('--up_k', type=int, default=10) # tnet的上层宽度
    # parser.add_argument('--up_l', type=int, default=20) # tnet的上层深度

    parser.add_argument('--regularized_skip_connection', type=bool, default=True, help='promote connections to closer nodes') 

    parser.add_argument('--boolean_hardness_aware_loss_1',type=bool, default=True, help='setup hardness aware loss part1 after acc=95')
    # parser.add_argument('--boolean_hardness_aware_loss_alpha', type=float, default=2)
    parser.add_argument('--boolean_hardness_aware_loss_2', type=bool, default=True, help='start hardness aware loss part2 after acc=99') # 
    parser.add_argument('--boolean_hardness_aware_loss_beta', type=int, default=10) 
    parser.add_argument('--descent_tau',type=bool, default=True)     

    parser.add_argument('--stop_at_100', type=bool, default=True, help='stop training when acc=100')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_model_freq', type=int, default=1000)
    parser.add_argument('--save_acc_threshold', type=bool, default=False, help='save results after acc=99')
    parser.add_argument('--log_tb', type=bool, default=True) # 必须true

    return parser.parse_args()

if __name__ == '__main__':

    ####################################################################################################################
    args = get_args()

    args.truthtable_name = os.path.splitext(args.truthtable_file)[0]
    
    args.dataset_dir = f'./truthtable/{args.truthtable_name}.truth'

    # get network size from net_config.py
    assert tnet_size.__contains__(args.truthtable_name), f'truthtable_name {args.truthtable_name} not in size dict. Please add size in net_config.py'
    args.up_k = tnet_size[args.truthtable_name]['width_up']
    args.up_l = tnet_size[args.truthtable_name]['depth_up']
    args.down_k = tnet_size[args.truthtable_name]['width_down']
    args.down_l = tnet_size[args.truthtable_name]['depth_down']

    time = datetime.datetime.now().strftime('%Y-%m-%d,%H:%M:%S')

    print(f'tb_log is {args.log_tb}')
    if not os.path.exists(f"./output/{args.truthtable_name}-{args.down_k}-{args.down_l}-{args.up_k}-{args.up_l}/"):
        os.makedirs(f"./output/{args.truthtable_name}-{args.down_k}-{args.down_l}-{args.up_k}-{args.up_l}/")
    args.save_dir = f"./output/{args.truthtable_name}-{args.down_k}-{args.down_l}-{args.up_k}-{args.up_l}/"

    writer = None
    if args.log_tb:
        tb_dir = f"./tb_log/{args.truthtable_name}-{args.down_k}-{args.down_l}/{args.seed}/{time}"
        args.tb_dir = tb_dir
        txt_dir = os.path.join(tb_dir, 'args.txt')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        writer = SummaryWriter(tb_dir)
        print(f'created writer, {tb_dir}')

    if args.save_acc_threshold:
        thresholds_to_save = [0.98,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999,0.9993,0.9995,0.9997]
        current_threshold_index = 0   

    ####################################################################################################################

    print(vars(args))
    # print('hear loss + adaptive to biggest loss, loss/ wrong rate')
    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader, test_set = load_dataset(args)
    model= get_model(args)
    loss_fn = torch.nn.MSELoss(reduce=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    node, _, init_gates, init_inverters = model.count_connected_node()
    list_of_gates = [init_gates]
    print(f'init node {node}, init inverters {init_inverters}')
    
    ####################################################################################################################

    best_acc = 0
    best_node = 1e10
    epoch = 0
    iteration = 0

    while iteration < args.num_iterations:
        torch.cuda.empty_cache()
        loss = 0

        loss = train(model, train_loader, loss_fn, optim, args, epoch, iteration = iteration, best_acc = best_acc)
        iteration += len(train_loader)

        if (epoch+1) % np.clip((args.eval_freq // len(train_loader)),1,None) == 0:
            torch.cuda.empty_cache()
            print("\nEpoch:",epoch+1)
            print("Iteration:",iteration)

            train_accuracy_eval_mode = eval(model, test_loader, False, args)

            print("loss:", loss)
            print('acc_eval_mode: ', train_accuracy_eval_mode)
            # caclulate the number of gates and levels
            node, lev, gates, invs = model.count_connected_node()
            print('total gates = ', gates[args.input_bits : -args.output_bits].sum().item(), 'lev = ', lev)

            if args.log_tb:
                writer.add_scalar('acc', train_accuracy_eval_mode, iteration)
                writer.add_scalar('loss/loss_acc', loss, iteration)
                writer.add_scalar('nodes/nodes', node, iteration)
                writer.add_scalar('nodes/lev', lev, iteration)
 
            if train_accuracy_eval_mode > best_acc:
                best_acc = train_accuracy_eval_mode
                print('IS THE BEST ACC UNTIL NOW.')

                if args.descent_tau and best_acc >= 0.99:
                    tau = 1.5-math.exp(70*(best_acc-1))
                    model.tau = tau
                    print(f"New tau = {tau}")

                if args.save_acc_threshold and args.save_model and best_acc >= thresholds_to_save[current_threshold_index]:
                    acc_str = int(best_acc * 1000)
                    # save_path = os.path.join(args.save_dir, f"{iteration}_{acc_str}.pth")
                    torch.save(model, os.path.join(args.save_dir, f"{iteration}_{acc_str}.pth"))
                    print(f"Save model")
                    if current_threshold_index < len(thresholds_to_save) - 1:
                        current_threshold_index += 1
                    else:
                        args.save_acc_threshold = False


        if best_acc == 1.0 and args.stop_at_100:
            print(f"Save model")
            torch.save(model, os.path.join(args.save_dir, f"{iteration}_100acc.pth"))
            print('Finished Training')
            exit()
        
        epoch+=1

    print('Finished Training')

    ####################################################################################################################



