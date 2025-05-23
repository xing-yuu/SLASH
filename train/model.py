
import torch
import torch.nn as nn
import numpy as np
import math
# from PH_NETFourier_feature_mapping import *
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from train._utils import reset_padding,boundary_padding
# from train.utils.point_transformer import GVAPatchEmbed
from train.multigrid import calculate_residual_2D,smooth_PCG_sp_assembly,calculate_residual,smooth_pcg_train,smooth_pcg_infer,calculate_legal_coordinates,node_KF_assembly,feature_assembly_M,coarse2fine_interpolation_error,fine2coarse_interpolation,calculate_level_K_list,calculate_adj_coo
import time

class RFFeatureMap(torch.nn.Module):
    def __init__(self, input_dim, output_dim, sigma=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.omega = nn.Parameter(torch.randn(output_dim, input_dim) * self.sigma,requires_grad=False)
        self.b = nn.Parameter(torch.rand(output_dim) * 2 * np.pi,requires_grad=False)
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        z = torch.cos(torch.mm(x, self.omega.t()) + self.b)
        return z


class Encoder(ME.MinkowskiNetwork):
    def __init__(self,CHANNELS = [36,64],D=3):
        super(Encoder, self).__init__(D)
        self.conv_1a=ME.MinkowskiConvolution(in_channels=CHANNELS[0],out_channels=CHANNELS[1],kernel_size=1,stride=1,dimension=D)
        self.bn_1=ME.MinkowskiBatchNorm(CHANNELS[1])
        self.FC=ME.MinkowskiLinear(CHANNELS[1], CHANNELS[1])
        self.relu=ME.MinkowskiReLU(inplace=True)

    def forward(self, x,coo_mapping):
        out=reset_padding(self.conv_1a(x),coo_mapping)
        out=self.bn_1(out)
        out=self.relu(out)
        out=reset_padding(self.relu(self.FC(out)),coo_mapping)
        return out

class Smoother(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,coo,U_model,k_model_list,f_model,smooth_it,batch_size):
        F_=[]
        for i in range (0,batch_size):
            mask=coo[:,0]==i
            f_=f_model[mask].reshape(-1,3,6).reshape(-1,6)
            U_=smooth_PCG_sp_assembly(k_model_list[i],f_,U_model[mask],smooth_it)
            F_.append(U_)
        return torch.cat(F_,dim=0)


class U_Net(ME.MinkowskiNetwork):
    

    def __init__(self, CHANNELS = [64,96,128], D=3):
        super(U_Net, self).__init__(D)
        
        self.Conv1 =nn.Sequential(
            ME.MinkowskiConvolution(in_channels=CHANNELS[0],out_channels=CHANNELS[1],kernel_size=3,stride=2,dimension=D),
            ME.MinkowskiConvolution(in_channels=CHANNELS[1],out_channels=CHANNELS[1],kernel_size=3,stride=1,dimension=D),
            ME.MinkowskiBatchNorm(CHANNELS[1]),
            ME.MinkowskiReLU(inplace=True),
            
        ) 
        
        self.Conv2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=CHANNELS[1],out_channels=CHANNELS[2],kernel_size=3,stride=2,dimension=D),
            ME.MinkowskiConvolution(in_channels=CHANNELS[2],out_channels=CHANNELS[2],kernel_size=3,stride=1,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            # ME.MinkowskiBatchNorm(CHANNELS[2])
        ) 

        self.Upsample_2 =nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=CHANNELS[2],out_channels=CHANNELS[1],kernel_size=3,stride=2,dimension=D),
            ME.MinkowskiConvolution(in_channels=CHANNELS[1],out_channels=CHANNELS[1],kernel_size=3,stride=1,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            # ME.MinkowskiBatchNorm(CHANNELS[1])
        ) 
        self.Up_conv2 =nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2*CHANNELS[1],out_channels=CHANNELS[1],kernel_size=3,stride=1,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            # ME.MinkowskiBatchNorm(CHANNELS[1])
        ) 

        self.Upsample_1 =nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=CHANNELS[1],out_channels=CHANNELS[0],kernel_size=3,stride=2,dimension=D),
            ME.MinkowskiConvolution(in_channels=CHANNELS[0],out_channels=CHANNELS[0],kernel_size=3,stride=1,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            # ME.MinkowskiBatchNorm(CHANNELS[0])
        ) 
        self.Up_conv1 =nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2*CHANNELS[0],out_channels=CHANNELS[0],kernel_size=3,stride=1,dimension=D),
            ME.MinkowskiBatchNorm(CHANNELS[0]),
            ME.MinkowskiReLU(inplace=True),
            # ME.MinkowskiBatchNorm(CHANNELS[0])
        ) 

    def forward(self, x,coo_mapping):
        x1 = self.Conv1(x) # C[0] -> C[1]
        x2 = self.Conv2(x1) # C[1] -> C[2]
        d  = ME.cat(x1,self.Upsample_2(x2)) # CAT(C[1],C[2]->C[1])
        d  = self.Up_conv2(d) # 2*C[1]-> C[1]
        d  = ME.cat(x ,self.Upsample_1(d))
        d  = self.Up_conv1(d)
        d = reset_padding(d,coo_mapping)
        return d



class Decoder(ME.MinkowskiNetwork):
    def __init__(self, CHANNELS = [128,64,18],D=3):
        super(Decoder, self).__init__(D)
        # print(CHANNELS[0],CHANNELS[1],D)
        self.conv_1a=ME.MinkowskiConvolution(in_channels=CHANNELS[0],out_channels=CHANNELS[1],kernel_size=3,stride=1,dimension=D)
        self.conv_1b=ME.MinkowskiConvolution(in_channels=CHANNELS[1],out_channels=CHANNELS[1],kernel_size=3,stride=1,dimension=D)
        self.bn=ME.MinkowskiBatchNorm(CHANNELS[1])
        self.relu=ME.MinkowskiReLU(inplace=True)
        self.FC=ME.MinkowskiLinear(CHANNELS[1], CHANNELS[2])
        self.RESTRICT=ME.MinkowskiTanh()
        self.smoother=Smoother()
    
    def boundary_condition(self,U_h):
        bc_list_of_featurs=[]
        list_of_featurs = U_h.decomposed_features
        for  fea in list_of_featurs:
            bc_list_of_featurs.append(fea-torch.mean(fea,dim=0,keepdim=True))
        return torch.cat(bc_list_of_featurs,dim=0)


    def forward(self,x ,coo_mapping, no_padding_coo,k_batch,f_batch,smooth_it,batch_size):
        out=self.conv_1a(x)
        out=reset_padding(out,coo_mapping)
        out=self.conv_1b(out)
        out=reset_padding(out,coo_mapping)
        out=self.bn(out)
        out=self.relu(out)
        out = self.FC(out)
        out=self.RESTRICT(out)
        out=reset_padding(out,coo_mapping)
        U_h_feature = out.features_at_coordinates(no_padding_coo.float())
        U_h= ME.SparseTensor(
                    coordinates=no_padding_coo.float(),
                    features=U_h_feature
                )
        # return U_h_feature
        
        list_of_featurs=self.boundary_condition(U_h)
        list_of_featurs=self.smoother(no_padding_coo,list_of_featurs,k_batch,f_batch,smooth_it,batch_size)
        return list_of_featurs
    # torch.cat(list_of_featurs,dim=0)
        # return 

class V_Net(ME.MinkowskiNetwork):
    def __init__(self,r,sub_net=None,D=3):
        super(V_Net, self).__init__(D)
        self.predict=U_Net([64,128,256])
        self.r=r
        self.sub_net=sub_net
        self.interpolation=ME.MinkowskiInterpolation()
    def forward(self,input_list,coo_mapping_list):
 
        U_imp_list=None
        fea_h_padding_imp=self.predict(input_list[0],coo_mapping_list[0])
        if self.sub_net is not None:
            U_imp_list=self.sub_net(input_list[1:],coo_mapping_list[1:])
            coo_h=fea_h_padding_imp.C.clone().float()
            coo_h[:,1:]/=2.0
            e_h_feature = self.interpolation(U_imp_list[len(U_imp_list)-1], coo_h)
            e=ME.SparseTensor(  e_h_feature,
                                coordinate_manager=fea_h_padding_imp.coordinate_manager, 
                                coordinate_map_key=fea_h_padding_imp.coordinate_map_key  
                            )
            fea_h_padding_imp=fea_h_padding_imp+e
        
        if U_imp_list is None :
            U_imp_list=[fea_h_padding_imp]
        else :
            U_imp_list.append(fea_h_padding_imp)
        return U_imp_list
    @torch.no_grad()
    def infer(self,input_list,coo_mapping_list):
        fea_h_padding_imp=self.predict(input_list[0],coo_mapping_list[0])
        if self.sub_net is not None:
            U_imp=self.sub_net.infer(input_list[1:],coo_mapping_list[1:])
            coo_h=fea_h_padding_imp.C.clone().float()
            coo_h[:,1:]/=2.0
            e_h_feature = self.interpolation(U_imp, coo_h)
            e=ME.SparseTensor(  e_h_feature,
                                coordinate_manager=fea_h_padding_imp.coordinate_manager, 
                                coordinate_map_key=fea_h_padding_imp.coordinate_map_key  
                            )
            fea_h_padding_imp=fea_h_padding_imp+e
        return fea_h_padding_imp
       
    
    
    
class ISINS(nn.Module):
    def __init__(self,cfg):
        super(ISINS, self).__init__()
        self.r=cfg.model.resolution
        self.level=cfg.model.level
        self.sub_net=None
        self.encoder=Encoder([36+32,64])
        self.position_encoder=RFFeatureMap(3,32)
        
        self.decoder=self.decoder=nn.ModuleList([Decoder([128,64,18]) for _ in range(self.level)])
        for i in range (cfg.model.level):
            self.sub_net=V_Net(r=self.r/(2**(cfg.model.level-i-1)),sub_net=self.sub_net)
 
    
    def forward(self,batch,level_coo_list,K_model_list,f_batch_list,smooth_it,batch_size):
        PE_level_coo=self.position_encoder((level_coo_list[0].float()[:,1:]))
        feature=feature_assembly_M(batch["coordinates"][:,0],batch["node_type"],batch["C_H"],batch["C_H_soft"])
        feature=torch.cat((feature,PE_level_coo),dim=1)
        feature.requires_grad_(True)
        coo_h,fea_h,coo_mapping_h=boundary_padding(level_coo_list[0],feature,self.r)
        fea_h_padding=ME.SparseTensor(coordinates=coo_h.float(),features=fea_h)
        fea_h_padding=self.encoder(fea_h_padding,coo_mapping_h)
        
        # INput list
        input_list=[fea_h_padding]
        coo_mapping_list=[coo_mapping_h]
        for i in range (1,self.level):
            input_2h=fine2coarse_interpolation(input_list[len(input_list)-1],level_coo_list[i],self.r/(2**i))
            coo_2h,fea_2h,coo_mapping_2h=boundary_padding(level_coo_list[i],input_2h,self.r/(2**i))
            input_list.append(ME.SparseTensor(coordinates=coo_2h.float(),features=fea_2h))
            coo_mapping_list.append(coo_mapping_2h)
        
        U_imp_list=self.sub_net(input_list,coo_mapping_list)
        U_imp_list.reverse()
        U_exp_list=[]
        # for decoder,U_imp,input,coo_map,level_coo,K_batch,f_batch in zip(self.decoder,U_imp_list,input_list,coo_mapping_list,level_coo_list,K_model_list,f_batch_list):
        #     U_exp_list.append(decoder(ME.cat(U_imp,input),coo_map,level_coo,K_batch,f_batch,smooth_it,batch_size))
        for i in range (self.level):
            U_exp_list.append(self.decoder[i](ME.cat(U_imp_list[i],input_list[i]),coo_mapping_list[i],level_coo_list[i],K_model_list[i],f_batch_list[i],smooth_it,batch_size))
        
        # U_exp_list[len(U_exp_list)-1]=self.smoother(level_coo_list[0], U_exp_list[len(U_exp_list)-1],K_model_list[0],f_batch_list[0],smooth_it,batch_size)
        
        
        return U_exp_list
    @torch.no_grad()
    def infer(self,batch,level_coo_list,K_model_list,f_batch_list,smooth_it,batch_size):
        PE_level_coo=self.position_encoder((level_coo_list[0].float()[:,1:]))
        feature=feature_assembly_M(batch["coordinates"][:,0],batch["node_type"],batch["C_H"],batch["C_H_soft"])
        feature=torch.cat((feature,PE_level_coo),dim=1)
        # feature.requires_grad_(True)
        coo_h,fea_h,coo_mapping_h=boundary_padding(level_coo_list[0],feature,self.r)
        fea_h_padding=ME.SparseTensor(coordinates=coo_h.float(),features=fea_h)
        fea_h_padding=self.encoder(fea_h_padding,coo_mapping_h)
        
        # INput list
        input_list=[fea_h_padding]
        coo_mapping_list=[coo_mapping_h]
        for i in range (1,self.level):
            input_2h=fine2coarse_interpolation(input_list[len(input_list)-1],level_coo_list[i],self.r/(2**i))
            coo_2h,fea_2h,coo_mapping_2h=boundary_padding(level_coo_list[i],input_2h,self.r/(2**i))
            input_list.append(ME.SparseTensor(coordinates=coo_2h.float(),features=fea_2h))
            coo_mapping_list.append(coo_mapping_2h)
        
        U_imp=self.sub_net.infer(input_list,coo_mapping_list)
        return self.decoder[0](ME.cat(U_imp,input_list[0]),coo_mapping_list[0],level_coo_list[0],K_model_list[0],f_batch_list[0],smooth_it,batch_size)
       
