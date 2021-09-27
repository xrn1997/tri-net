import os

import torch

import params
import tools.utils as ut
from trains.train import Trainer


def main():
    print("main")
    # 准备数据集
    train_dataloader = ut.get_dataloader(params.dataset)
    test_dataloader = ut.get_dataloader(params.dataset, train=False)

    # 初始化模块
    feature_extractor = params.feature_extractor_dict[params.dataset]
    label_predictor = params.label_predictor_dict[params.dataset]

    # 初始化优化器
    optimizer = torch.optim.SGD([{'params': feature_extractor.parameters()},
                                 {'params': label_predictor[0].parameters()},
                                 {'params': label_predictor[1].parameters()},
                                 {'params': label_predictor[2].parameters()}], lr=params.learning_rate, momentum=0.9)

    # 加载训练参数
    save_path = params.tri_net_save_path
    if not os.path.exists(params.save_dir):
        os.mkdir(params.save_dir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(save_path + "/fe.pth"):
        feature_extractor.load_state_dict(torch.load(save_path + "/fe.pth"))
    if os.path.exists(save_path + "/lp1.pth"):
        label_predictor[0].load_state_dict(torch.load(save_path + "/lp1.pth"))
        label_predictor[1].load_state_dict(torch.load(save_path + "/lp2.pth"))
        label_predictor[2].load_state_dict(torch.load(save_path + "/lp3.pth"))

    # 初始化Trainer
    trainer = Trainer(feature_extractor, label_predictor, optimizer)
    for epoch in range(params.epochs):
        trainer.train(epoch=epoch, dataloader=train_dataloader)
        trainer.test(dataloader=test_dataloader)


if __name__ == '__main__':
    main()
