import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def bmc_loss(pred, target, noise_var):
    """
    Compute the Balanced MSE Loss (BMC) between 'pred' and the ground truth 'target'.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    #logits = - (pred - target.T).pow(2) / (2 * noise_var) # logit size: [batch, batch]
    logits = - (pred - target.permute(*torch.arange(target.ndim - 1, -1, -1))).pow(2) / (2 * noise_var)
    #print('noise_var:', float(noise_var.detach()))
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).float()) # contrastive-like loss
    loss = loss * (2 * noise_var).detach() # optional: restore the loss scale, 'detach' when noise is learnable 
    return loss, noise_var

class BMCLoss(torch.nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)