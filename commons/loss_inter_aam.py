#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class AAMINTER(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False,num_center=3,top_k=5,margin_negative=0.06, **kwargs):
        super(AAMINTER, self).__init__()
        self.top_k=top_k 
        self.mask = [margin_negative,] * top_k + [0,] * int((nClasses-top_k))
        self.mask = torch.tensor(self.mask).reshape(1,-1).float()
        self.test_normalize = True
        self.num_center=num_center
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses * num_center, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        
        # cos(theta)
        cosine_all = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine_all = cosine_all.view(x.size()[0], -1, self.num_center)
        cosine, _ = torch.max(cosine_all, dim=2)
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)        
        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        sorted_cosine=cosine.detach() 
        sorted_cosine=(one_hot * -1.) + (1.0-one_hot) *  sorted_cosine
        sorted_cosine, indices = torch.sort(sorted_cosine,descending=True)
        mask = torch.tensor(self.mask).reshape(1,-1).float().repeat(one_hot.size(0),1)
        mask=mask.to(sorted_cosine.device)
        one_hot_ = torch.zeros_like(sorted_cosine)
        one_hot_.scatter_(1, indices,mask )

        cosine = cosine + one_hot_

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1