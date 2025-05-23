import torch
import math
import numpy as np
import MinkowskiEngine as ME
from train.multigrid import node_KF_assembly,displacement_regularization,assembly_sp

interpolation = ME.MinkowskiInterpolation()
# class MPELoss_with_backward(torch.autograd.Function):

class MPELoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx,F,node_stiffness,node_load,coo,batch_size):

        with torch.no_grad():
            ctx.save_for_backward(F)
            energy, grad= MPE_full_all_batch(F,node_stiffness,node_load,coo,batch_size)
            ctx.gradient = grad
            output = energy
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input, voxel, Ke_hard, Fe_hard,Ke_soft,Fe_soft = ctx.saved_variables
        grad = ctx.gradient

        return grad, None, None, None, None,None,None

@torch.no_grad()
def MPE_full_all_batch(U,K,F,coo,batch_size):
    grad=[]
    energy=0
    for i in range (0,batch_size):
        mask=coo[:,0]==i
        u_=U[mask].reshape(-1,6,3).transpose(1,2).reshape(-1,6)
        f_=F[mask].view(-1,3,6).view(-1,6)
        ku =torch.sparse.mm(K[i], u_)
        grad.append(ku-f_)
        energy+=torch.sum(torch.mm(torch.mm(u_.t(),ku*0.5-f_).view(6,6),torch.eye(6, 6, device=U.device)))

    return energy/batch_size,torch.cat(grad,dim=0).reshape(-1,3,6).transpose(1,2).reshape(-1,18)





def MPE_loss_all_layer(optimizer,K_list,U_list,F_list,coo_list,batch_size,weight):
    

    loss_dict={}
    # MPE
    loss_func=MPELoss.apply
    loss=torch.tensor(0.0,requires_grad=True,device=U_list[0].device)

    # loss_U=torch.tensor(0.0,requires_grad=True,device=device)
    for i in range(len(U_list)):
        # print(torch.max(F_list[i]),torch.min(F_list[i]))
        # print(torch.max(U_list[i]),torch.min(U_list[i]))
        level_loss=loss_func(U_list[i],K_list[i],F_list[i],coo_list[i],batch_size)
        loss=loss+level_loss
        loss_dict["loss level: "+str(i)] = level_loss.item()

    loss_dict["loss"] = loss.item()
    optimizer.zero_grad()
    # torch.autograd.set_detect_anomaly(True)
    loss.backward()
    optimizer.step()
    return loss_dict