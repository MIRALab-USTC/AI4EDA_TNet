import time
import numpy as np
from torch.utils.data import Dataset
import torch
import math
import os


import random

import random

class TruthTableDataset(Dataset):
    def __init__(self, random_data=True, input_nums=4, truth_table_file=None, truth_flip=False):
        super(TruthTableDataset, self).__init__()
        self.input_nums = input_nums
        self.truth_flip = truth_flip
        if random_data:
            self.labels = self.random_generate_truth_table(self.input_nums)
            if self.input_nums<9:
                self.data = self.generate_inputs(self.labels)
            else:
                self.data = self.generate_binary_array(self.labels)

        else:
            self.labels = self.read_data_from_file(truth_table_file)
            self.data = self.generate_binary_array(self.labels)
            
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    # def read_data_from_file(self, filename):
    #     try:
    #         with open(filename, 'r') as file:
    #             lines = file.readlines()
    #             data = []
    #             for line in lines:
    #                 line_data = [int(x) for x in line.strip().split()]
    #                 data.append(line_data)
    #             numpy_array = np.array(data)
    #             return numpy_array.T
    #     except Exception as e:
    #         print(f"读入文件错误: {e}")
    #         return None

    def read_data_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
                data = []
                for line in lines:
                    # Check if the line contains spaces
                    if ' ' in line:
                        line_data = [int(x) for x in line.strip().split()]
                    else:
                        line_data = [int(char) for char in line.strip()]
                    data.append(line_data)
                numpy_array = np.array(data)
                return numpy_array.T
        except Exception as e:
            print(f"读入文件错误: {e}")
            return None

    
    def generate_inputs(self, truth_table_number): #对于n＜8
        '''
        input: truth table number, e.g. 1010
        output: x 00
                01
                10
                11
                y 1010
        '''
        length = len(truth_table_number)
        bit_size = int(math.log2(length))
        x = np.arange(length).astype(np.uint8)[:, None]
        x = np.unpackbits(x, axis = 1)[:, -bit_size:]
        return x


    def random_generate_truth_table(self, bit_num): 
        '''
        input: truth table bits, e.g. 2
        output: y 1010
        '''
        y = np.random.randint(0, 2, size = 2**bit_num)
        print(y)
        # filename='16.turth'
        # np.savetxt(filename, y,fmt='%d',delimiter=None,newline='')

        return y.T
    

    def generate_binary_array(self, labels): #对于n＜16
        result = []
        length = len(labels)
        bit_size = int(math.log2(length))
        for i in range(length):
            row = [int(x) for x in bin(i)[2:].zfill(bit_size)]
            result.append(row)
            
        if self.truth_flip:
            a = np.flip(np.array(result),axis=(0,1))
            # return np.flip(a,axis=1)
            return a.copy()

        return np.array(result)

    


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break
        
    



if __name__ == '__main__':

    bc_train_set = TruthTableDataset(random_data=True, input_nums=4)
    print('\ntruth table')
    print(bc_train_set.data.shape)
    print(bc_train_set.labels.shape)
    train_loader = torch.utils.data.DataLoader(bc_train_set, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)
    loader = iter(train_loader)

    print(bc_train_set.__len__())
    for i, (x, y) in enumerate(load_n(train_loader, 100)):
        pass


