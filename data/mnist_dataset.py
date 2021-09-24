import os

import numpy as np
import torch
from torch import multiprocessing
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import params


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
                temp = []
                for k in range(3):
                    noise = torch.relu(torch.randn(10) * params.std)
                    noise[j[1]] = noise[j[1]] + 1
                    noise = noise / noise.sum()
                    temp.append(noise)
                self.labels.append(temp)
            np.save(label_save_path, self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.labels[index][0], self.labels[index][1], self.labels[index][2]


#  测试用代码
if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])
    dataset = MNISTDataSet(root='./MNIST', transform=tf)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=params.batch_size,  # 每次处理的batch大小
                            shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                            num_workers=multiprocessing.cpu_count(),  # 线程数
                            pin_memory=True)
    for i in dataloader:
        print(i)
        break

