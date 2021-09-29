import numpy as np
import torch.cuda
from torch.autograd import Variable
from logzero import logger
import models
import params
from tools import utils
from tools.utils import labeling, des


class Trainer:
    def __init__(self, feature_extractor, label_predictor: [], optimizer):
        if torch.cuda.is_available() and params.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.fe = feature_extractor.to(self.device)
        self.lp1 = label_predictor[0].to(self.device)
        self.lp2 = label_predictor[1].to(self.device)
        self.lp3 = label_predictor[2].to(self.device)

        # 损失函数
        self.class_criterion = models.OneHotNLLLoss(reduction='multiply')
        self.tri_net_criterion = models.TriNetLoss()
        # 优化器
        self.optimizer = optimizer

    def train(self, epoch, dataloader):
        # setup models
        self.fe.train()
        self.lp1.train()
        self.lp2.train()
        self.lp3.train()
        # steps
        start_steps = epoch * len(dataloader)
        total_steps = params.initial_epochs * len(dataloader)
        for batch_idx, data in enumerate(dataloader):
            p = float(batch_idx + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-params.gamma * p)) - 1
            inputs, labels = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            label1 = Variable(labels[:, 0]).to(self.device, non_blocking=True)
            label2 = Variable(labels[:, 1]).to(self.device, non_blocking=True)
            label3 = Variable(labels[:, 2]).to(self.device, non_blocking=True)
            # 优化器
            self.optimizer = utils.optimizer_scheduler(self.optimizer, p)
            self.optimizer.zero_grad()
            # 提取特征
            feature = self.fe(inputs)
            # 预测标签
            preds1 = self.lp1(feature)
            preds2 = self.lp2(feature)
            preds3 = self.lp3(feature)
            # 损失
            loss1 = self.class_criterion(preds1, label1)
            loss2 = self.class_criterion(preds2, label2)
            loss3 = self.class_criterion(preds3, label3)
            # 总损失
            loss = self.tri_net_criterion(loss1, loss2, loss3)
            # 反向传播
            loss.backward()
            self.optimizer.step()
            # 每5批次输出一次损失
            if (batch_idx + 1) % 5 == 0:
                logger.info('epoch:{}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    loss.item()
                ))
        save_path = params.tri_net_save_path
        torch.save(self.fe.state_dict(), save_path + "/fe.pth")
        torch.save(self.lp1.state_dict(), save_path + "/lp1.pth")
        torch.save(self.lp2.state_dict(), save_path + "/lp2.pth")
        torch.save(self.lp3.state_dict(), save_path + "/lp3.pth")

    def test(self, dataloader):
        # setup models
        self.fe.eval()
        self.lp1.eval()
        self.lp2.eval()
        self.lp3.eval()
        # 设置变量
        correct1 = 0.0
        correct2 = 0.0
        correct3 = 0.0
        for batch_idx, data in enumerate(dataloader):
            inputs, label = data

            inputs = Variable(inputs).to(self.device, non_blocking=True)
            label = Variable(label).to(self.device, non_blocking=True)

            # 提取特征
            feature = self.fe(inputs)
            # 预测标签
            pred1 = self.lp1(feature).data.max(1, keepdim=True)[1]
            pred2 = self.lp2(feature).data.max(1, keepdim=True)[1]
            pred3 = self.lp3(feature).data.max(1, keepdim=True)[1]

            correct1 += pred1.eq(label.data.view_as(pred1)).cpu().sum()
            correct2 += pred2.eq(label.data.view_as(pred1)).cpu().sum()
            correct3 += pred3.eq(label.data.view_as(pred1)).cpu().sum()

        logger.info('\n预测器1的正确率: {}/{} ({:.4f}%)'.format(
            correct1, len(dataloader.dataset), 100. * float(correct1) / len(dataloader.dataset)
        ))
        logger.info('\n预测器2的正确率: {}/{} ({:.4f}%)'.format(
            correct2, len(dataloader.dataset), 100. * float(correct2) / len(dataloader.dataset)
        ))
        logger.info('\n预测器3的正确率: {}/{} ({:.4f}%)'.format(
            correct3, len(dataloader.dataset), 100. * float(correct3) / len(dataloader.dataset)
        ))

    def update(self, mu) -> None:
        """
        :param mu:  未标记的数据集
        """
        # setup models
        self.fe.train()
        self.lp1.train()
        self.lp2.train()
        self.lp3.train()
        md = [self.lp1, self.lp2, self.lp3]
        flag = 1
        sigma = params.sigma_0
        for t in range(1, params.T + 1):
            n_t = min(1000 * 2 ^ t, params.U)
            if n_t == params.U:
                if t % 4 == 0:
                    continue
            if flag == 1:
                flag = 0
                sigma_t = sigma - params.sigma_os
            else:
                sigma_t = sigma

            for v in range(1, 4):
                plv = []
                plv = labeling(self.fe, md[(v + 1) % 3], md[(v + 1) % 3], mu, n_t, sigma_t)
                plv = des(self.fe, plv, md[(v + 1) % 3], md[(v + 1) % 3])
                if v == 1:
                    for epoch in range(params.update_epochs):
                        self.train(epoch=epoch, )
                else:
                    for epoch in range(params.update_epochs):
                        self.train(epoch=epoch, )
