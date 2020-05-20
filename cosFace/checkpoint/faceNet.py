import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import code


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 0.35):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features) )
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / torch.clamp(xlen.view(-1,1) * wlen.view(1,-1), min=1e-8 )
        cos_theta = cos_theta.clamp(-1,1)

        # IMPLEMENT phi_theta
        phi_theta = cos_theta - self.m
        
#         cos_theta = cos_theta * xlen.view(-1,1)
#         phi_theta = phi_theta * xlen.view(-1,1)

        output = (cos_theta,phi_theta)
        return output


class CustomLoss(nn.Module):
    def __init__(self, s=64 ):
        super(CustomLoss, self).__init__()
        self.s = s

    def forward(self, input, target):
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        # IMPLEMENT loss
        
        #numerator
        numerator = torch.exp(self.s * phi_theta[torch.arange(phi_theta.size(0)), target[:,0]])        

        #denominator
        denominator = torch.exp(self.s * cos_theta)
        denominator = denominator.sum(dim=1) - denominator[torch.arange(denominator.size(0)), target[:,0]]
        denominator = denominator + numerator
        
        #loss
        loss = torch.mean(torch.log(numerator / denominator )).neg()
        

        _, predictedLabel = torch.max(cos_theta.data, 1)
        predictedLabel = predictedLabel.view(-1, 1)
        accuracy = (predictedLabel.eq(target.data).cpu().sum().item() ) / float(target.size(0) )

        return loss, accuracy
    
class faceNet(nn.Module):
    def __init__(self,classnum=10574, feature=False, m = 1.35):
        super(faceNet, self).__init__()
        self.classnum = classnum
        self.feature = feature

        # IMPLEMENT resdiual network 20-layer with batch normalization

        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = CustomLinear(in_features = 512,
                out_features = self.classnum, m=m)


    def forward(self, x):
        
        # IMPLEMENT forward of network

        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x) )
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x) )
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x) ) ) )

        x = x.view(x.size(0),-1)
        x = self.fc5(x)

        if self.feature:
            return x

        x = self.fc6(x)
        return x

class faceNet_BN(nn.Module):
    def __init__(self,classnum=10574, feature=False, m = 1.35):
        super(faceNet_BN, self).__init__()
        self.classnum = classnum
        self.feature = feature

        # IMPLEMENT resdiual network 20-layer with batch normalization

        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.conv1_bn_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.PReLU(64)
        
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.conv1_bn_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.PReLU(64)
        
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.conv1_bn_3 = nn.BatchNorm2d(64)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.conv2_bn_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.PReLU(128)
        
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.conv2_bn_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.PReLU(128)
        
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.conv2_bn_3 = nn.BatchNorm2d(128)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.conv2_bn_4 = nn.BatchNorm2d(128)
        self.relu2_4 = nn.PReLU(128)
        
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.conv2_bn_5 = nn.BatchNorm2d(128)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.conv3_bn_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.PReLU(256)
        
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.conv3_bn_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.PReLU(256)
        
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.conv3_bn_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.conv3_bn_4 = nn.BatchNorm2d(256)
        self.relu3_4 = nn.PReLU(256)
        
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.conv3_bn_5 = nn.BatchNorm2d(256)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.conv3_bn_6 = nn.BatchNorm2d(256)
        self.relu3_6 = nn.PReLU(256)
        
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.conv3_bn_7 = nn.BatchNorm2d(256)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.conv3_bn_8 = nn.BatchNorm2d(256)
        self.relu3_8 = nn.PReLU(256)
        
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.conv3_bn_9 = nn.BatchNorm2d(256)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.conv4_bn_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.PReLU(512)
        
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.conv4_bn_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.PReLU(512)
        
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.conv4_bn_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = CustomLinear(in_features = 512,
                out_features = self.classnum, m=m)


    def forward(self, x):
        
        # IMPLEMENT forward of network

        x = self.relu1_1(self.conv1_bn_1(self.conv1_1(x)))
        x = x + self.relu1_3(self.conv1_bn_3(self.conv1_3(self.relu1_2(self.conv1_bn_2(self.conv1_2(x))))))
        

        x = self.relu2_1(self.conv2_bn_1(self.conv2_1(x)))
        x = x + self.relu2_3(self.conv2_bn_3(self.conv2_3(self.relu2_2(self.conv2_bn_2(self.conv2_2(x))))))
        x = x + self.relu2_5(self.conv2_bn_5(self.conv2_5(self.relu2_4(self.conv2_bn_4(self.conv2_4(x))))))

        x = self.relu3_1(self.conv3_bn_1(self.conv3_1(x) ))
        x = x + self.relu3_3(self.conv3_bn_3(self.conv3_3(self.relu3_2(self.conv3_bn_2(self.conv3_2(x))))))
        x = x + self.relu3_5(self.conv3_bn_5(self.conv3_5(self.relu3_4(self.conv3_bn_4(self.conv3_4(x))))))
        x = x + self.relu3_7(self.conv3_bn_7(self.conv3_7(self.relu3_6(self.conv3_bn_6(self.conv3_6(x))))))
        x = x + self.relu3_9(self.conv3_bn_9(self.conv3_9(self.relu3_8(self.conv3_bn_8(self.conv3_8(x))))))

        x = self.relu4_1(self.conv4_bn_1(self.conv4_1(x) ))
        x = x + self.relu4_3(self.conv4_bn_3(self.conv4_3(self.relu4_2(self.conv4_bn_2(self.conv4_2(x) ) )) ))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)

        if self.feature:
            return x

        x = self.fc6(x)
        return x