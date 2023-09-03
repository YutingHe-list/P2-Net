# -*- coding: utf-8 -*-

import torch

from torch import nn
from torch.nn.functional import max_pool3d
import numpy as np

class dice_coef(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1.
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        a = torch.sum(y_true * y_pred, (2, 3, 4))
        b = torch.sum(y_true, (2, 3, 4))
        c = torch.sum(y_pred, (2, 3, 4))
        dice = (2 * a) / (b + c + smooth)
        return torch.mean(dice)

class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):

        return torch.mean((y_true - y_pred) ** 2)

class MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class partical_MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, mask, Lambda=0.5):

        return torch.mean(torch.abs(y_true - y_pred) * mask) * Lambda

class mix_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):

        return crossentry()(y_true, y_pred) + 1 - dice_coef()(y_true, y_pred)


class crossentry(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred+smooth))
class B_crossentry(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # print(y_true.size())
        # print(y_pred.size())
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred+smooth)+(1-y_true)*torch.log(1-y_pred+smooth))

class HRA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, label, alpha=0.1):
        smooth = 1e-6
        w = torch.abs(label - predict)
        w = torch.where(w > alpha, torch.full_like(predict, 1), torch.full_like(predict, 0))
        loss_ce = -torch.sum(w * label * torch.log(predict + smooth), dim=(2, 3, 4)) / torch.sum(w + smooth, dim=(2, 3, 4))
        return torch.mean(loss_ce)

class DropOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, label, drop_rate=0.1):
        smooth = 1e-6
        # 计算L1距离
        l1 = torch.abs(label - predict)
        w = torch.where(l1 > drop_rate, torch.full_like(predict, 1), torch.full_like(predict, 0))
        # 生成概率矩阵


        w = torch.where(w > 0, torch.full_like(predict, 1), torch.full_like(predict, 0))
        # predict(np.sum(w.data.cpu().numpy()))
        loss_ce = -torch.sum(w * label * torch.log(predict + smooth)) / torch.sum(w + smooth)
        return loss_ce


class cox_regression_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, time):
        smooth = 1e-6

        events = y_true.view(-1)
        log_h = y_pred.view(-1)
        time = time.view(-1)
        order = torch.argsort(time)
        events = events[order]
        log_h = log_h[order]
        log_h_max = log_h.data.max()

        hazard_ratio = torch.exp(log_h-log_h_max)
        log_risk = torch.log(torch.cumsum(hazard_ratio,dim=0)+ smooth)+log_h_max
        uncensored_likelihood = log_h - log_risk
        censored_likelihood = uncensored_likelihood * events
        neg_likelihood = -torch.sum(censored_likelihood)
        num_events = torch.sum(events)
        loss = neg_likelihood / (num_events + smooth)
        return loss


