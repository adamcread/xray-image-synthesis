import torch
from torch import nn

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5, threshold=0.95):
        # binarize inputs and targets for segmentation loss
        inputs = 1 - torch.sigmoid(1e3*(inputs-threshold))
        targets = 1 - torch.sigmoid(1e3*(targets-threshold))
        
        intersection = (inputs * targets).sum()  
        union = inputs.sum() + targets.sum() - intersection

        IoU_loss = 1 - (intersection + smooth)/(union + smooth) 
        
        return IoU_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets, smooth=1e-5, threshold=0.95): 
        # binarize inputs and targets for segmentation loss  
        inputs = 1 - self.sigmoid(1e3*(inputs-threshold))
        targets = 1 - self.sigmoid(1e3*(targets-threshold)) 

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice_loss = 1 - (2*intersection + smooth)/(union + smooth)  
        
        return dice_loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.BCELoss = nn.BCELoss()
    
    def forward(self, inputs, targets, smooth=1e-5, threshold=0.95):
        # binarize inputs and targets for segmentation loss  
        inputs = 1 - torch.sigmoid(1e3*(inputs-threshold))
        targets = 1 - torch.sigmoid(1e3*(targets-threshold))

        bce_loss = self.BCELoss(inputs, targets)

        return bce_loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.BCELoss = nn.BCELoss()

    def forward(self, inputs, targets, threshold=0.95, alpha=0.8, gamma=2):
        inputs = 1 - torch.sigmoid(1e3*(inputs-threshold))
        targets = 1 - torch.sigmoid(1e3*(targets-threshold))  
        
        bce_loss = self.BCELoss(inputs, targets)
        focal_loss = alpha * (1-torch.exp(-bce_loss))**gamma * bce_loss
                       
        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5, threshold=0.95, alpha=0.5, beta=0.5):
        inputs = 1 - torch.sigmoid(1e3*(inputs-threshold))
        targets = 1 - torch.sigmoid(1e3*(targets-threshold))   
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        tversky_loss = 1- (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return tversky_loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.BCELoss = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1e-5, threshold=0.95):
        # binarize inputs and targets for segmentation loss  
        inputs = 1 - torch.sigmoid(1e3*(inputs-threshold))
        targets = 1 - torch.sigmoid(1e3*(targets-threshold))
        
        intersection = (inputs * targets).sum()  
        union = inputs.sum() + targets.sum()

        dice_loss = 1 - (2*intersection + smooth)/(union + smooth)  
        bce_loss = self.BCELoss(inputs, targets)

        dice_bce_loss = (bce_loss + dice_loss)/2
        
        return dice_bce_loss


class DiceFocalLoss(nn.Module):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()
        self.BCELoss = nn.BCELoss()
    
    def forward(self, inputs, targets, threshold=0.95, smooth=1e-5, alpha=0.8, gamma=2):
        inputs = 1 - torch.sigmoid(1e3*(inputs-threshold))
        targets = 1 - torch.sigmoid(1e3*(targets-threshold))

        bce_loss = self.BCELoss(inputs, targets)
        focal_loss = alpha * (1-torch.exp(-bce_loss))**gamma * bce_loss

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice_loss = 1 - (2*intersection + smooth)/(union + smooth)

        dice_focal_loss = (focal_loss + dice_loss)/2

        return dice_focal_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.BCELoss = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1e-5, alpha=0.5, beta=0.5, gamma=1, threshold=0.95):
        inputs = 1 - torch.sigmoid(1e3*(inputs-threshold))
        targets = 1 - torch.sigmoid(1e3*(targets-threshold))
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        tversky_loss = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        focal_tversky_loss = (1 - tversky)**gamma
                       
        return focal_tversky_loss