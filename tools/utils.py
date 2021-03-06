import time

import logzero
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import params
import custom_dataset


def get_dataset(dataset, train=True):
    """
     获取dataset

    :param dataset: 数据集名称
    :param train: 是否是训练集
    :return: dataset
    """
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])
        if train:
            data = custom_dataset.MNISTDataSet(root=params.mnist_path, transform=transform)
        else:
            data = datasets.MNIST(root=params.mnist_path, train=False, transform=transform,
                                  download=True)
    else:
        raise Exception('There is no dataset1 named {}'.format(str(dataset)))
    return data


def get_dataloader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=False):
    """
    dataloader的一层封装

    :param dataset: 数据集
    :param batch_size: batch大小
    :param shuffle: 是否乱序
    :param drop_last: 不足batch大小时是否丢弃
    :return: 返回dataloader
    """
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,  # 每次处理的batch大小
                      shuffle=shuffle,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                      num_workers=1,  # 线程数
                      pin_memory=False,
                      drop_last=drop_last)


def optimizer_scheduler(optimizer, p):
    """
    调整学习率

    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer


def gen_labels(label: int, types=10, num=3) -> list:
    """
    根据标签生成n个添加噪声的one-hot标签

    :param num: 生成的标签数量
    :param label: 原始标签，int类型
    :param types: 分类数量
    :return: 添加噪声后的标签列表,该列表中有3个one-hot标签向量（列表）
    """
    temp = []
    for k in range(num):
        noise = torch.relu(torch.randn(types) * params.std)
        noise[label] = noise[label] + 1
        noise = noise / noise.sum()
        temp.append(noise.tolist())
    return temp


def log_save(save_dir, start_time=time.time(), limit=0):
    """
    日志存储

    :param save_dir: 存储目录
    :param start_time: 开始时间
    :param limit: 隔多久新开一个文件，单位为秒
    """
    if time.time() - start_time > limit:
        logzero.logfile(save_dir + "/output_" + str(time.time()) + ".log")
        start_time = time.time()
