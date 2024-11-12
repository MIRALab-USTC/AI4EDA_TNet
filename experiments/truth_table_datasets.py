import numpy as np
from torch.utils.data import Dataset
import torch
import math


class TruthTableDataset(Dataset):
    def __init__(self, input_nums=4, truth_table_file=None, truth_flip=False):
        super(TruthTableDataset, self).__init__()
        self.input_nums = input_nums
        self.truth_flip = truth_flip

        self.labels = self.read_data_from_file(truth_table_file)
        self.data = self.generate_binary_array(self.labels)
            
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

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
            print(f"Read file error: {e}")
            return None

    def generate_binary_array(self, labels): 
        result = []
        length = len(labels)
        bit_size = int(math.log2(length))
        for i in range(length):
            row = [int(x) for x in bin(i)[2:].zfill(bit_size)]
            result.append(row)
            
        if self.truth_flip:
            a = np.flip(np.array(result),axis=(0,1))
            return a.copy()

        return np.array(result)


if __name__ == '__main__':
    dataset_dir = '../truthtable/example.txt'
    bc_train_set = TruthTableDataset(input_nums=2, truth_table_file = dataset_dir, truth_flip = True)
    print('\ntruth table')
    print(bc_train_set.data.shape)
    print(bc_train_set.labels.shape)
    train_loader = torch.utils.data.DataLoader(bc_train_set, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)

    print(bc_train_set.__len__())


