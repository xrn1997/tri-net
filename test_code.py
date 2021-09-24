"""
测试代码块的地方，尝试各种函数，与程序本体无关。
"""
import torch.nn as nn
import torch

if __name__ == '__main__':
    std = 0.001
    relu = nn.ReLU()
    result = relu(torch.randn(3, 3) * std)
    print(result)
    print(result.sum())
    print(result/result.sum())

    print(torch.zeros(3))

