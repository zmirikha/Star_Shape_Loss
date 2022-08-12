import torch
from torch.autograd import Variable
import torch.nn as nn
from reg_conv import gen_b




def total_loss(input, target, ng):
    """ 
    Compute the total loss

    Args:
    input (torch.Tensor): predicted segmentation probability map
    target (torch.Tensor): binary groundtruth segmentation map
    ng (torch.Tensor): 1-hot encoded neighbouring maps)
    """
    alpha = 1  # CE loss adjustment factor
    beta = 40  # star shape loss adjustment factor
    
    l1 = nn.BCELoss()
    l_sh = star_loss(input, target, ng)
    
    return (l1(input, target) * alpha) + (l_sh * beta)

def star_loss(input, target, ng):
    """ 
    Compute the star shape loss

    Args:
    input (torch.Tensor): predicted segmentation probability map
    target (torch.Tensor): binary groundtruth segmentation map
    ng (torch.Tensor): 1-hot encoded neighbouring maps)
    """
    
    [dim_0, dim_1] = input.size()

    inp = input.unsqueeze(0)
    tar = target.unsqueeze(0)
    # one hot coded neighbouring maps.

    # Generate a mask of B for each GT mask as defined in the paper's eq. 5.
        
    b = gen_b(tar) # Convolve gt with regional kernels.
        
    tmp = torch.sum(torch.mul(ng, b), 0)  # use the right kernel for each region by one-hot encoded regional maps.   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_eq = Variable(torch.Tensor(1, 1, dim_0,dim_1).zero_().to(device))

    b = torch.eq(tmp.unsqueeze(0), z_eq).float()  # compute B_{pq}

    # Compute if two pixels on a line segment are assigned a same label.
        
    prob_diff = gen_b(inp)  # Convolve predicted map with regional kernels.
   
    diff = torch.sum(torch.mul(ng, torch.abs(prob_diff)), 0).unsqueeze(0)  # use the right kernel for each region by one-hot encoded regional maps. 
    
    abs_err = torch.abs(inp.sub(tar.unsqueeze(0)))  #the difference of predicted probabilities and ground truth
    
    # Compute the final star shape loss defined in the paper's eq. 4.
    loss_per_image = torch.mean(torch.mul(torch.mul(b, diff), abs_err))
    
    return loss_per_image
