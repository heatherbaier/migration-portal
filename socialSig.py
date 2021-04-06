import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import torch
import math

import helpers
importlib.reload(helpers)

from helpers import *

# import socialSig
# importlib.reload(socialSig)

class bilinearImputation(torch.nn.Module):
    '''
    Class to create the social signature image
    '''
    def __init__(self, X):
        super(bilinearImputation, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.arange(0, X.shape[1]), dtype = torch.float32, requires_grad=True))
        # self.outDim = [10,10]
        self.outDim = [224,224]
        self.inDim = math.ceil(math.sqrt(X.shape[1]))

    def forward(self, batchX):
        # print("    W at beginning: ", torch.tensor(self.W, dtype = torch.int)) 
        taken = torch.take(batchX, construct_indices(torch.clamp(torch.tensor(self.W, dtype = torch.int64), 0, 29), batchX.shape[0], self.W.shape[0]))
        batchX.data = batchX.data.copy_(taken.data)        
        inDataSize = self.W.shape[0] #Data we have per dimension
        targetSize = self.inDim ** 2
        paddingOffset = targetSize - inDataSize
        paddedInX = torch.nn.functional.pad(input=batchX, pad=(0,paddingOffset), mode="constant", value=0)
        buildImage = torch.reshape(paddedInX,(batchX.shape[0], 1, self.inDim, self.inDim))   
        return torch.nn.functional.interpolate(buildImage, size=([self.outDim[0], self.outDim[1]]), mode='bilinear')


class bilinearImputationNoDrop(torch.nn.Module):
    '''
    Class to create the social signature image
    '''
    def __init__(self, X):
        super(bilinearImputationNoDrop, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.random.rand(X.shape[1])*.1, dtype = torch.float32, requires_grad=True))
        # self.outDim = [10,10]
        self.outDim = [224,224]
        self.inDim = math.ceil(math.sqrt(X.shape[1]))

    def forward(self, batchX):
        # print("    W at beginning: ", torch.tensor(self.W)) 
        taken = torch.take(batchX, construct_noOverlap_indices(torch.tensor(self.W, dtype = torch.float32), batchX.shape[0], self.W.shape[0]))
        batchX.data = batchX.data.copy_(taken.data)   

        inDataSize = self.W.shape[0] #Data we have per dimension
        targetSize = self.inDim ** 2
        paddingOffset = targetSize - inDataSize
        paddedInX = torch.nn.functional.pad(input=batchX, pad=(0,paddingOffset), mode="constant", value=0)
        buildImage = torch.reshape(paddedInX,(batchX.shape[0], 1, self.inDim, self.inDim))   

        return torch.nn.functional.interpolate(buildImage, size=([self.outDim[0], self.outDim[1]]), mode='bilinear')

###### Define our model
class SocialSigNet(torch.nn.Module):
    '''
    SocialSigNet
    Mocks the ResNet18 architecture
    '''
    def __init__(self, X, outDim):
        super().__init__()
        self.SocialSig = bilinearImputationNoDrop(X=X)                
        self.conv2d = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxPool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)#,
            # torch.nn.Sequential(
            #     torch.nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
            #     torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            # )
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.seqBlock1 = torch.nn.Sequential(self.block1, self.block1)
        self.seqBlock2 = torch.nn.Sequential(self.block2, self.block3)
    

        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   (downsample): Sequential(
            #     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            #     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   )
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.seqBlock3 = torch.nn.Sequential(self.block4, self.block5)

        self.block6 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   (downsample): Sequential(
            #     (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            #     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   )
            )
        self.block7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )


        self.seqBlock4 = torch.nn.Sequential(self.block6, self.block7)

        # 512*outDim
        # self.linear = torch.nn.Linear(512*outDim, outDim)
        self.linear = torch.nn.Linear(in_features=512*7*7, out_features=1)
        

        
    def forward(self, X, epoch):
        
        out = self.SocialSig(X) # OUT:  torch.Size([100, 1, 10, 10])
        #print(out.shape)
        #print(out[0])
        #print("====")
        #print(out[1])
        # print('Imputed s512*uccessfully')
        # print(out.shape)

        pd.DataFrame(out.clone()[0].flatten()).to_csv("./figs2/im" + str(epoch) + ".csv")
        out = self.conv2d(out)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxPool(out)
        out = self.seqBlock1(out)
        out = self.seqBlock2(out)
        out = self.seqBlock3(out)
        out = self.seqBlock4(out)
        out = self.relu(out)

        out = out.flatten(start_dim=1)

        out = self.linear(out)
        return out