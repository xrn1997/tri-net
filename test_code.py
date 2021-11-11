"""
测试代码块的地方，尝试各种函数，与程序本体无关。
"""
import math
import numpy
import torch.nn as nn
import torch
from logzero import logger
from torch.utils.data import DataLoader

import custom_dataset
import params


def test_list2dataset(lists, batch_size=4):
    """
    测试列表.

    :param lists: [[data1,label1],[data2,label2],....]
    :return:
    """
    dataset = custom_dataset.List2DataSet(lists)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,  # 每次处理的batch大小
                            shuffle=False,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                            num_workers=1,  # 线程数
                            pin_memory=True)
    logger.info(len(dataset))
    for index, data in enumerate(dataloader):
        inputs, _ = data
        logger.info(inputs.shape)


def test_ne():
    """
    测试 ne.

    :return:
    """
    a = torch.Tensor([1])
    b = torch.Tensor([1])
    logger.info(a.ne(b))


def test_relu():
    """
     测试 relu.

    :return:
    """

    std = 0.001
    relu = nn.ReLU()
    result = relu(torch.randn(3, 3) * std)
    logger.info(result)
    logger.info(result.sum())
    logger.info(result / result.sum())


def test_divide():
    """
    测试 /和%区别.
    """
    logger.info(3.5 / 2)
    logger.info(3.5 % 2)
    logger.info(1 % 3)


def test_unsqueeze():
    """
    测试reshape。
    """
    a = torch.Tensor([[1, 2], [1, 3]])
    logger.info(a.shape)
    logger.info(a.unsqueeze(dim=0).shape)


def test_sum():
    a = torch.Tensor([False, True, False, False])
    b = torch.Tensor([[False], [False], [False], [False]])
    logger.info(b)
    logger.info(b.squeeze())
    c = torch.zeros(4)
    logger.info(c)
    c = c + a + b.squeeze()
    logger.info(c)


if __name__ == '__main__':
    # 测试列表
    plv = [[torch.Tensor([1, 2]), torch.Tensor([2])], [torch.Tensor([2, 3]), torch.Tensor([3])]]
    test_list2dataset(plv)
    test = numpy.load(params.tri_net_save_path + "/l0_1.npy", allow_pickle=True)
    test_list2dataset(test, batch_size=2)
