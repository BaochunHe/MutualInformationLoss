import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np

def thirdOrderSplineKernel(u):
    abs_u = u.abs()
    sqr_u = abs_u.pow(2.0)

    result = torch.FloatTensor(u.size()).zero_()
    result = result.to(u.device)

    mask1 = abs_u<1.0
    mask2 = (abs_u>=1.0 )&(abs_u<2.0)

    result[mask1] = (4.0 - 6.0 * sqr_u[mask1] + 3.0 * sqr_u[mask1] * abs_u[mask1]) / 6.0
    result[mask2] = (8.0 - 12.0 * abs_u[mask2] + 6.0 * sqr_u[mask2] - sqr_u[mask2] * abs_u[mask2]) / 6.0

    return result

class MILoss(nn.Module):
    def __init__(self, num_bins=16):
        super(MILoss, self).__init__()
        self.num_bins = num_bins


    def forward(self, moving, fixed):
        moving = moving.view(moving.size(0), -1)
        fixed = fixed.view(fixed.size(0), -1)

        padding = float(2)
        batchsize = moving.size(0)

        fixedMin = fixed.min(1)[0].view(batchsize,-1)
        fixedMax = fixed.max(1)[0].view(batchsize,-1)

        movingMin = moving.min(1)[0].view(batchsize,-1)
        movingMax = moving.max(1)[0].view(batchsize,-1)
        #print(fixedMax,movingMax)

        JointPDF = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()
        movingPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        fixedPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        JointPDFSum = torch.FloatTensor(batchsize).zero_()
        JointPDF_norm = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()

        if JointPDF.device != moving.device:
            JointPDF = JointPDF.to(moving.device)
            movingPDF = movingPDF.to(moving.device)
            fixedPDF = fixedPDF.to(moving.device)
            JointPDFSum = JointPDFSum.to(moving.device)
            JointPDF_norm = JointPDF_norm.to(moving.device)

        #print(JointPDF.device)

        fixedBinSize = (fixedMax - fixedMin) / float((self.num_bins - 2 * padding))
        movingBinSize = (movingMax - movingMin) / float(self.num_bins - 2 * padding)

        fixedNormalizeMin = fixedMin / fixedBinSize - float(padding)
        movingNormalizeMin = movingMin / movingBinSize - float(padding)

        #print(fixed.shape,fixedBinSize.shape,fixedNormalizeMin.shape)
        fixed_winTerm = fixed / fixedBinSize - fixedNormalizeMin

        fixed_winIndex = fixed_winTerm.int()
        fixed_winIndex[fixed_winIndex < 2] = 2
        fixed_winIndex[fixed_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        moving_winTerm = moving / movingBinSize - movingNormalizeMin

        moving_winIndex = moving_winTerm.int()
        moving_winIndex[moving_winIndex < 2] = 2
        moving_winIndex[moving_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        for b in range(batchsize):
            a_1_index = moving_winIndex[b] - 1
            a_2_index = moving_winIndex[b]
            a_3_index = moving_winIndex[b] + 1
            a_4_index = moving_winIndex[b] + 2

            a_1 = thirdOrderSplineKernel((a_1_index - moving_winTerm[b]))
            a_2 = thirdOrderSplineKernel((a_2_index - moving_winTerm[b]))
            a_3 = thirdOrderSplineKernel((a_3_index - moving_winTerm[b]))
            a_4 = thirdOrderSplineKernel((a_4_index - moving_winTerm[b]))
            for i in range(self.num_bins):
                fixed_mask = (fixed_winIndex[b] == i)
                fixedPDF[b][i] = fixed_mask.sum()
                for j in range(self.num_bins):
                    JointPDF[b][i][j] = a_1[fixed_mask & (a_1_index == j)].sum() + a_2[
                        fixed_mask & (a_2_index == j)].sum() + a_3[fixed_mask & (a_3_index == j)].sum() + a_4[
                                            fixed_mask & (a_4_index == j)].sum()

            #JointPDFSum[b] = JointPDF[b].sum()
            #norm_facor = 1.0 / JointPDFSum[b]
            #print(JointPDF[b])
            JointPDF_norm[b] = JointPDF[b] / JointPDF[b].sum()
            fixedPDF[b] = fixedPDF[b] / fixed.size(1)

        movingPDF = JointPDF_norm.sum(1)
        
        #print(JointPDF_norm)

        MI_loss = torch.FloatTensor(batchsize).zero_().to(moving.device)
        for b in range(batchsize):
            JointPDF_mask = JointPDF_norm[b] > 0
            movingPDF_mask = movingPDF[b] > 0
            fixedPDF_mask = fixedPDF[b] > 0

            MI_loss[b] = (JointPDF_norm[b][JointPDF_mask] * JointPDF_norm[b][JointPDF_mask].log()).sum() \
                         - (movingPDF[b][movingPDF_mask] * movingPDF[b][movingPDF_mask].log()).sum() \
                         - (fixedPDF[b][fixedPDF_mask] * fixedPDF[b][fixedPDF_mask].log()).sum()

        #print(MI_loss)
        loss = MI_loss.mean()
        return -1.0*loss