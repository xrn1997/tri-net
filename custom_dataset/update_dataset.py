import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from logzero import logger
import params
import custom_dataset


class UpdateDataSet(Dataset):
    """
    给数据集套一层壳
    """

    def __init__(self, old_data):
        self.data = old_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]


# 测试
if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])
    dataset1 = custom_dataset.MNISTDataSet(root='./MNIST', transform=tf)
    # dataloader = DataLoader(dataset=dataset1,
    #                         batch_size=2,  # 每次处理的batch大小
    #                         shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
    #                         num_workers=1,  # 线程数
    #                         pin_memory=True)
    # for loader in dataloader:
    #     data, label = loader
    #     logger.info(data)
    #     logger.info(label)
    #     break

    first_size, second_size = 1, len(dataset1) - 1
    first_dataset, second_dataset = torch.utils.data.random_split(dataset1, [first_size, second_size])
    third_size, forth_size = 1, len(second_dataset) - 1
    third_dataset, forth_dataset = torch.utils.data.random_split(second_dataset, [third_size, forth_size])

    dataset2 = UpdateDataSet(first_dataset)
    logger.info(dataset2[0])
    dataset3 = UpdateDataSet(third_dataset)
    logger.info(dataset3[0])
    dataset4 = dataset2 + dataset3
    logger.info(len(dataset4))
    for i in dataset4:
        logger.info(i)