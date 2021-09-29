import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import params
from tools import utils


class MNISTDataSet(Dataset):
    def __init__(self, root, transform=None):
        self.data = datasets.MNIST(root=root, train=True, transform=transform,
                                   download=True)
        # output smearing 输出涂抹，给标签加白噪声
        label_save_path = root + "/labels.npy"
        self.labels = []
        if os.path.exists(label_save_path):
            self.labels = np.load(label_save_path, allow_pickle=True)
        else:
            for j in self.data:
                temp = utils.gen_labels(j[1])
                self.labels.append(temp)
            np.save(label_save_path, self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], torch.Tensor(self.labels[index])


#  测试用代码
if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])
    dataset = MNISTDataSet(root='./MNIST', transform=tf)
    length = len(dataset)
    first_size, second_size = params.initial_size, length - params.initial_size
    first_dataset, second_dataset = torch.utils.data.random_split(dataset, [first_size, second_size])

    print(len(first_dataset))
    print(len(second_dataset))
    dataloader = DataLoader(dataset=first_dataset,
                            batch_size=16,  # 每次处理的batch大小
                            shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                            num_workers=1,  # 线程数
                            pin_memory=True)

    time_start = time.time()
    for i in dataloader:
        print(i)
        break
    time_end = time.time()
    print('MNIST DataSet totally cost', time_end - time_start)