from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import params
from data.mnist_dataset import MNISTDataSet


class UpdateDataSet(Dataset):
    """
    给数据集套一层壳
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self[index][1]


if __name__ == '__main__':
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])
    dataset = MNISTDataSet(root='./MNIST', transform=tf)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=2,  # 每次处理的batch大小
                            shuffle=True,  # shuffle的作用是乱序，先顺序读取，再乱序索引。
                            num_workers=1,  # 线程数
                            pin_memory=True)
    for loader in dataloader:
        data, label = loader
        print(data)
        print(label)
        break
