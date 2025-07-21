import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
from diffusion import KDLoss

########### model_path  ###########

# path to a plain resnet50 model
res50_path = '../ResNet/resnet50-0676ba61.pth'

########### model_path  ###########

class dsd_res50(nn.Module):
    def __init__(self, pretrained=True):
        super(dsd_res50, self).__init__()
        resnet = models.resnet50()
        if pretrained:
            resnet.load_state_dict(torch.load(res50_path))

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.fc = resnet.fc

        self.adpool = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Linear(2048, 45)

        self.fc1 = nn.Linear(256, 45)
        self.fc2 = nn.Linear(512, 45)
        self.fc3 = nn.Linear(1024, 45)

        self.diff1 = KDLoss(45, 45)
        self.diff2 = KDLoss(45, 45)
        self.diff3 = KDLoss(45, 45)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        output1 = x
        feature1 = x
        x = self.layer2(x)
        output2 = x
        feature2 = x
        x = self.layer3(x)
        output3 = x
        feature3 = x
        x = self.layer4(x)

        output1 = self.adpool(output1).view(output1.size(0), output1.size(1))
        output2 = self.adpool(output2).view(output2.size(0), output2.size(1))
        output3 = self.adpool(output3).view(output3.size(0), output3.size(1))
        output1 = self.fc1(output1)
        output2 = self.fc2(output2)
        output3 = self.fc3(output3)

        x1 = self.adpool(x).view(x.size(0), x.size(1))
        x1 = self.fc0(x1)
        gt_pred = x1.data
        gt_output3 = output3.data
        gt_output2 = output2.data

        diff_loss1, stu_fea3, tea_fea4 = self.diff1(gt_pred, output3)
        diff_loss2, stu_fea2, tea_fea3 = self.diff2(gt_output3, output2)
        diff_loss3, stu_fea1, tea_fea2 = self.diff3(gt_output2, output1)

        gt_tea_fea4 = tea_fea4.data
        gt_tea_fea3 = tea_fea3.data
        gt_tea_fea2 = tea_fea2.data

        # return x1, output1, output2, output3, gt_pred, gt_output2, gt_output3, diff_loss1, diff_loss2, diff_loss3
        return x1, stu_fea1, stu_fea2, stu_fea3, gt_pred, gt_output2, gt_output3, diff_loss1, diff_loss2, diff_loss3
