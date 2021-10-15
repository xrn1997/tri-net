"""
测试代码块的地方，尝试各种函数，与程序本体无关。
"""
import torch.nn as nn
import torch
from logzero import logger

if __name__ == '__main__':
    std = 0.001
    relu = nn.ReLU()
    result = relu(torch.randn(3, 3) * std)
    logger.info(result)
    logger.info(result.sum())
    logger.info(result / result.sum())
    logger.info(torch.zeros(3))
    logger.info(3.5 / 2)
    logger.info(3.5 % 2)
    logger.info(1 % 3)

