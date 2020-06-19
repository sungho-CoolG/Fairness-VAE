"""ops.py"""

import torch
import torch.nn.functional as F


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z,t_dim,c_dim,p_dim):
    assert z.dim() == 2
   
    B, _ = z.size()
    perm_z_list = []
    z_j=z.split(1,1)
    perm = torch.randperm(B).to(z.device)
    perm_c=torch.randperm(B).to(z.device)
    for j in range(int(z.size(1))):
        if j<int(z.size(1)-p_dim-c_dim):
            perm_z_j = z_j[j]
            perm_z_list.append(perm_z_j)
        
        elif j<int(z.size(1)-p_dim):
            perm_z_j = z_j[j][perm_c]
            perm_z_list.append(perm_z_j)
        else:
            
            perm_z_j = z_j[j][perm]
            perm_z_list.append(perm_z_j)
    perm_z=torch.cat(perm_z_list, 1)
    
    return perm_z

'''
def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
'''