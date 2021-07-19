# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:57:01 2017

@author: Biagio Brattoli
"""
import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import torchvision.models as models

import sys
sys.path.append('Utils')

class Network(nn.Module):

    def __init__(self, classes=1000):
        super(Network, self).__init__()

        res50_model = models.resnet50(pretrained=True)        
        self.feature_extractor = nn.Sequential(*list(res50_model.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc1',nn.Linear(18432, 4096))
        self.classifier.add_module('relu1',nn.ReLU(inplace=True))
        self.classifier.add_module('drop1',nn.Dropout(p=0.05))
        self.classifier.add_module('fc2',nn.Linear(4096, 2048))
        self.classifier.add_module('relu2',nn.ReLU(inplace=True))
        self.classifier.add_module('fc3',nn.Linear(2048, classes))

    def save(self,checkpoint):
        res50_model = models.resnet50(pretrained=True)
        pretrained_last_layers = nn.Sequential(*list(res50_model.children())[-1:])
        res50_model_finetuned = nn.Sequential([self.feature_extractor, pretrained_last_layers])
        torch.save(res50_model_finetuned.state_dict(), checkpoint)
    
    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.feature_extractor(x[i])
            z = z.view([B,1,-1])
            x_list.append(z)

        x = cat(x_list,1)
        x = x.view(B,-1)
        x = self.classifier(x)

        return x
