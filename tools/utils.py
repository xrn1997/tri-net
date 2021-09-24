from torch import multiprocessing
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import params
from data.mnist_dataset import MNISTDataSet


def get_dataloader(dataset, train=True):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])
        if train:
            data = MNISTDataSet(root=params.mnist_path, transform=transform)
        else:
            data = datasets.MNIST(root=params.mnist_path, train=False, transform=transform,
                                  download=True)
        dataloader = DataLoader(dataset=data,
                                batch_size=params.batch_size,  # 每次处理的batch大小
                                shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                                num_workers=multiprocessing.cpu_count(),  # 线程数
                                pin_memory=True)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))
    return dataloader


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
