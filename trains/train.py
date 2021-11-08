import numpy as np
import torch.cuda
from torch.autograd import Variable
from logzero import logger
import models
import params
from tools import utils


class Trainer:
    def __init__(self, feature_extractor, label_predictor: [], optimizer):
        if torch.cuda.is_available() and params.use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.fe = feature_extractor.to(self.device)
        self.lp = [label_predictor[0].to(self.device), label_predictor[1].to(self.device),
                   label_predictor[2].to(self.device)]

        # 损失函数
        self.class_criterion = models.OneHotNLLLoss(reduction='multiply')
        self.tri_net_criterion = models.TriNetLoss()
        # 优化器
        self.optimizer = optimizer

    def train(self, epoch, dataset, mv=-1, ms=True):
        # 设置模式
        self.fe.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()

        dataloader = utils.get_dataloader(dataset=dataset)

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
            if ms:
                feature = self.fe(inputs)
            else:
                feature = inputs

            if mv == -1:
                # 预测标签
                preds1 = self.lp[0](feature)
                preds2 = self.lp[1](feature)
                preds3 = self.lp[2](feature)
                # 损失
                loss1 = self.class_criterion(preds1, label1)
                loss2 = self.class_criterion(preds2, label2)
                loss3 = self.class_criterion(preds3, label3)
                # 总损失
                loss = self.tri_net_criterion(loss1, loss2, loss3)
                # 反向传播
                loss.backward()
            else:
                # 预测标签
                preds = self.lp[mv](feature)
                # 损失
                loss = torch.nn.NLLLoss(preds)
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
        torch.save(self.lp[0].state_dict(), save_path + "/lp1.pth")
        torch.save(self.lp[1].state_dict(), save_path + "/lp2.pth")
        torch.save(self.lp[2].state_dict(), save_path + "/lp3.pth")

    def test(self, dataset):

        # 设置模式
        self.fe.eval()
        self.lp[0].eval()
        self.lp[1].eval()
        self.lp[2].eval()

        # 设置变量
        correct1 = 0.0
        correct2 = 0.0
        correct3 = 0.0

        dataloader = utils.get_dataloader(dataset=dataset)

        for batch_idx, data in enumerate(dataloader):
            inputs, label = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            label = Variable(label).to(self.device, non_blocking=True)

            # 提取特征
            feature = self.fe(inputs)

            # 预测标签
            pred1 = self.lp[0](feature).data.max(1, keepdim=True)[1]
            pred2 = self.lp[1](feature).data.max(1, keepdim=True)[1]
            pred3 = self.lp[2](feature).data.max(1, keepdim=True)[1]

            # 结果
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

    def update(self, initial_dataset, unlabeled_dataset) -> None:
        """
        更新模型。

        :param initial_dataset:  初始数据集。
        :param unlabeled_dataset:   未标记的数据集。
        """
        # setup models
        self.fe.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()

        flag = 1
        sigma = params.sigma_0
        mu = unlabeled_dataset
        lv = initial_dataset

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

            for v in range(0, 3):
                plv = self.labeling((v + 1) % 3, (v + 2) % 3, mu, n_t, sigma_t)
                plv = self.des(plv, (v + 1) % 3, (v + 2) % 3)
                lv = lv + plv
                if v == 0:
                    for epoch in range(params.update_epochs):
                        self.train(epoch=epoch, dataset=lv, mv=0)
                else:
                    for epoch in range(params.update_epochs):
                        self.train(epoch=epoch, dataset=lv, mv=v, ms=False)

    def labeling(self, mj, mh, mu, nt, sigma_t):
        """
        预测标签。两个预测器结果一致，并且平均最大后验概率大于σ_t.

        :param mj:  三个预测model之一。
        :param mh:  三个预测model之一，且与mj不同。
        :param mu: 未标记的数据集
        :param nt:  数据集大小（从mu中拿nt大小的未标记数据）。
        :param sigma_t: 过滤不确定的伪标签的阈值参数。
        :return: 打好伪标签的数据集。
        """
        dataloader = utils.get_dataloader(dataset=mu)

        plv = mu

        for batch_idx, data in enumerate(dataloader):
            inputs, _ = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)

            feature = self.fe(inputs)

            preds_j = self.lp[mj](feature).data.max(1, keepdim=True)
            preds_h = self.lp[mh](feature).data.max(1, keepdim=True)

            for i in range(0, params.batch_size):
                if preds_j[i][1] == preds_h[i][1] and (preds_j[i][0] + preds_h[i][0]) / 2 >= torch.log(sigma_t):
                    logger.debug("TODO 将数据合并到训练集中")

        return plv

    def des(self, plv, mj, mh):
        """
        判断标签是否稳定。

        :param plv: 待处理的有伪标签的数据集
        :param mj:  三个预测model之一。
        :param mh:  三个预测model之一，且与mj不同。
        :return: 稳定的伪标签数据集。
        """
        # 设置模式
        self.fe.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()

        dataloader = utils.get_dataloader(dataset=plv)

        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            label_j = Variable(labels[:, mj]).to(self.device, non_blocking=True)
            label_h = Variable(labels[:, mh]).to(self.device, non_blocking=True)

            # 记录预测错误的次数
            k = torch.zeros(params.batch_size)
            for time in range(0, 9):
                feature = self.fe(inputs)
                preds_j = self.lp[mj](feature)
                preds_h = self.lp[mh](feature)
                error_j = params.batch_size - preds_j.eq(label_j.data.view_as(label_j)).cpu().sum()
                error_h = params.batch_size - preds_h.eq(label_h.data.view_as(label_h)).cpu().sum()
                k = k + error_j + error_h

        return plv
