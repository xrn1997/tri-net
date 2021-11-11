import math
import os

import numpy
import numpy as np
import torch.cuda
from torch.autograd import Variable
from logzero import logger
import models
import params
import custom_dataset
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
        self.update_criterion = models.OneHotNLLLoss()
        self.tri_net_criterion = models.TriNetLoss()
        # 优化器
        self.optimizer = optimizer

    def train(self, epoch, dataset, mv=-1):
        logger.info("train")
        # 设置模式
        self.fe.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()
        # dataloader
        dataloader = utils.get_dataloader(dataset=dataset, drop_last=True)
        # steps
        start_steps = epoch * len(dataloader)
        total_steps = params.initial_epochs * len(dataloader)

        for batch_idx, data in enumerate(dataloader):

            p = float(batch_idx + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-params.gamma * p)) - 1

            # 优化器
            self.optimizer = utils.optimizer_scheduler(self.optimizer, p)
            self.optimizer.zero_grad()

            inputs, labels = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            if mv == -1:
                # 提取特征
                feature = self.fe(inputs)
                # 预测标签
                preds1 = self.lp[0](feature)
                preds2 = self.lp[1](feature)
                preds3 = self.lp[2](feature)
                # 标签
                label1 = Variable(labels[:, 0]).to(self.device, non_blocking=True)
                label2 = Variable(labels[:, 1]).to(self.device, non_blocking=True)
                label3 = Variable(labels[:, 2]).to(self.device, non_blocking=True)
                # 损失
                loss1 = self.class_criterion(preds1, label1)
                loss2 = self.class_criterion(preds2, label2)
                loss3 = self.class_criterion(preds3, label3)
                # 总损失
                loss = self.tri_net_criterion(loss1, loss2, loss3)
                # 反向传播
                loss.backward()
            else:
                # 提取特征
                feature = self.fe(inputs).data
                # 预测标签
                preds = self.lp[mv](feature)
                # 标签
                labels = Variable(labels).to(self.device, non_blocking=True)
                # 损失
                loss = self.update_criterion(preds, labels)
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
        logger.info("test")
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
        logger.info("update")
        # setup models
        self.fe.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()
        flag = 1
        sigma = params.sigma_0
        mu = unlabeled_dataset
        l = [[], [], []]
        for i in initial_dataset:
            l[0].append([i[0], i[1][0]])
            l[1].append([i[0], i[1][1]])
            l[2].append([i[0], i[1][2]])
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
                lv = l[v] + plv
                dataset = custom_dataset.List2DataSet(lv)
                self.train(epoch=params.update_epochs, dataset=dataset, mv=v)

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
        logger.info("labeling")
        dataloader = utils.get_dataloader(dataset=mu, shuffle=False)

        plv = []

        for batch_idx, data in enumerate(dataloader):
            if batch_idx * params.batch_size > nt:
                break
            inputs, _ = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)

            feature = self.fe(inputs)

            preds_j = self.lp[mj](feature).data.max(1, keepdim=True)
            preds_h = self.lp[mh](feature).data.max(1, keepdim=True)

            for i in range(0, params.batch_size):
                equals = (preds_j[1][i] == preds_h[1][i])
                confident = ((preds_j[0][i] + preds_h[0][i]) / 2 >= math.log(sigma_t))
                if equals and confident:
                    # 10分类，所以生成了一个长度为10的one-hot
                    label = torch.zeros(10)
                    label[preds_j[1][i].item()] = label[preds_j[1][i].item()] + 1
                    plv.append([inputs[i].cpu(), label])
        return plv

    def des(self, plv, mj, mh):
        """
        判断标签是否稳定。

        :param plv: 待处理的有伪标签的数据集
        :param mj:  三个预测model之一。
        :param mh:  三个预测model之一，且与mj不同。
        :return: 稳定的伪标签数据集。
        """
        logger.info("des")
        # 设置模式
        self.fe.train()
        self.lp[0].train()
        self.lp[1].train()
        self.lp[2].train()

        dataset = custom_dataset.List2DataSet(plv)
        dataloader = utils.get_dataloader(dataset, shuffle=False, drop_last=True)

        for index, data in enumerate(dataloader):
            inputs, labels = data
            inputs = Variable(inputs).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device, non_blocking=True).data.max(1, keepdim=True)[1]
            # 记录预测错误的次数
            k = torch.zeros(inputs.shape[0])
            for time in range(0, 9):
                feature = self.fe(inputs)
                preds_j = self.lp[mj](feature).data.max(1, keepdim=True)[1]
                preds_h = self.lp[mh](feature).data.max(1, keepdim=True)[1]
                error_j = preds_j.ne(labels.data.view_as(preds_j)).cpu().squeeze()
                error_h = preds_h.ne(labels.data.view_as(preds_h)).cpu().squeeze()
                k = k + error_j + error_h
            for n in range(0, inputs.shape[0]):
                if k[n] > 3:
                    plv.pop(index + n)

        return plv
