import os
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import argparse
from omegaconf import OmegaConf

from tqdm import tqdm
# from datagen import isotropic_elastic_tensor
from homogenization_helper import *
from numerical_homogenization import *
from node import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device='cpu'

def dis_vis(model,r): 
    model=model.reshape(6,3,r,r,r)
    for i in range(0,6):
        U=model[i]
        U=torch.linalg.norm(U,dim=0)
        # print(torch.max(U))
        # exit()
        U=(U / 0.8).detach()
        coo=torch.nonzero(U, as_tuple=False)
        U=U.cpu().numpy()
        coo=coo.detach().cpu().numpy()
        cmap = cm.get_cmap('jet',coo.shape[0])
        color= cmap(U[coo[:,0],coo[:,1],coo[:,2]])
        with open(str(r)+'-'+str(i)+'.obj', 'w') as f:
            for j in range (0,coo.shape[0]):
                print('v ',coo[j][0],coo[j][1],coo[j][2],
                    color[j][0],color[j][1],color[j][2],file=f)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
torch.set_printoptions(
    precision=4,    # ç²¾åº¦ï¼ä¿çå°æ°ç¹åå ä½ï¼é»è®¤4
    threshold=200,
    edgeitems=100,
    linewidth=150,  # æ¯è¡æå¤æ¾ç¤ºçå­ç¬¦æ°ï¼é»è®¤80ï¼è¶è¿åæ¢è¡æ¾ç¤º
    profile=None,
    sci_mode=True  # ç¨ç§å­¦ææ¯æ³æ¾ç¤ºæ°æ®ï¼é»è®¤True
)
if __name__=='__main__': 
    '''
    
    Args:
            Ke: [shape: 24*24], Element stiffness matrix 
            Fe: [shape: 24*6]: Element load

        Returns:
            K_node: [shape: 256*3*81], Node stiffness matrix 
            F_node: [shape: 256*3*6], Node load

    '''
    

    # output dir
    # if not os.path.exists(cfg.out_dir):
    #     os.makedirs(cfg.out_dir)

    # voxel_dir=os.path.join(cfg['dataset']['voxel']['dir'],'voxel.npz')

    cache_file='datagen/cache/fem_cache.npz'
    homo_helper=homogenization_helper(cache_file,device)

        
    
    res=64
    # res=128
    
    numerical_solver=numerical_homogenization(res,res,res,1,1,1,device)

    dataname='/mnt/local/project/dataset/single-0313/T-P-3-11-2.csv'
  
    E=1
    E_soft=E*1e-6
    v=0.3
    # v=random.uniform(0.2,0.4)
    C_H=isotropic_elastic_tensor(E,v).to(device)
    C_H_soft=isotropic_elastic_tensor(E_soft,v).to(device)
    X0,Ke,Fe=homo_helper.voxel_XKF(res,res,res,C_H)
    # X0,Ke,Fe=homo_helper.voxel_XKF(64,64,64,C_H)
    X0_soft,Ke_soft,Fe_soft=homo_helper.voxel_XKF(res,res,res,C_H_soft)
    K_node,F_node=assembly_node_matrix(Ke,Fe)
    K_node_soft,F_node_soft=assembly_node_matrix(Ke_soft,Fe_soft)
    # data = np.load(os.path.join(input_dir,dataname))  
    data = np.loadtxt(open(dataname,"rb"),dtype=np.int32)
    voxel = torch.from_numpy(data.astype(np.int32)).to(device).reshape(64,64,64)
    # voxel_ = torch.zeros(128,128,128,device=device)
    # base_coo=torch.nonzero(voxel, as_tuple=False)
    # hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
    # for i in hex:
    #     coo=(base_coo*2+i)%res
    #     voxel_[coo[:, 0], coo[:, 1], coo [:, 2]] = 1
    # voxel=voxel_.clone()
    
    # voxel_ = torch.zeros(256,256,256,device=device)
    # base_coo=torch.nonzero(voxel, as_tuple=False)
    # hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
    # for i in hex:
    #     coo=(base_coo*2+i)%res
    #     voxel_[coo[:, 0], coo[:, 1], coo [:, 2]] = 1
    # voxel=voxel_.clone()
    
    U=numerical_solver.solve_by_torch(voxel,Ke,Fe)
    dis_vis(U,res)
    print('U',torch.max(U),torch.min(U),torch.mean(U))
    CH_GT=numerical_solver.homogenized(voxel,U,Ke,X0)
    print('CH\n',CH_GT)
    