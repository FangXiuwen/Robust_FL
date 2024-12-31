import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

eps = 1e-7

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
   
class SelfAdaptiveTrainingSCE(torch.nn.Module):
    def __init__(self, num_classes=10, alpha=1, beta=0.3, temp=1, mu=1, com_epoch=0, total_epoch=40):
        super(SelfAdaptiveTrainingSCE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.temp = temp
        self.mu = mu
        self.com_epoch = com_epoch
        self.total_epoch = total_epoch

    def forward(self, logits, targets):
        # if epoch < self.es:
        #     return F.cross_entropy(logits, targets)
        label_one_hot = torch.nn.functional.one_hot(targets, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        # obtain prob, then update running avg
        prob = F.softmax(logits / self.temp, dim=1)
        w = self.com_epoch / (self.mu * self.total_epoch + self.com_epoch)
        soft_labels = (1 - w) * label_one_hot + w * prob.detach()
        # use symmetric cross entropy loss, without reduction
        loss = - self.alpha * torch.sum(soft_labels * torch.log(prob), dim=-1) \
                - self.beta * torch.sum(prob * torch.log(soft_labels), dim=-1)
        # sample weighted mean
        loss = loss.mean()
        return loss