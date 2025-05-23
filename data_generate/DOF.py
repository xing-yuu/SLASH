import os
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import argparse
from omegaconf import OmegaConf

from tqdm import tqdm
# from datagen import isotropic_elastic_tensor
# from homogenization_helper import *
# from numerical_homogenization import *
# from node import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__=='__main__': 
    '''
    
    Args:
            Ke: [shape: 24*24], Element stiffness matrix 
            Fe: [shape: 24*6]: Element load

        Returns:
            K_node: [shape: 256*3*81], Node stiffness matrix 
            F_node: [shape: 256*3*6], Node load

    '''
   
    #arguments
    parser=argparse.ArgumentParser(description="Dataset generator")
    parser.add_argument('--config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg=OmegaConf.load(args.config)

    # output dir
    # if not os.path.exists(cfg.out_dir):
    #     os.makedirs(cfg.out_dir)

    # voxel_dir=os.path.join(cfg['dataset']['voxel']['dir'],'voxel.npz')
    res=cfg.voxel.resolution

    


        

    for i in range (0,len(cfg.voxel.input_dir)):
        DOF=np.zeros(3,dtype=np.int64)
        voxel_nbr=0
        input_dir=cfg.voxel.input_dir[i]
        output_dir=cfg.out_dir[i]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        datanames = os.listdir(input_dir)
        datanames = random.sample(datanames,1000)
        length=len(datanames)
        tips=0
        CH_res=np.zeros((length,6,6),dtype=np.float32)
        pbar = tqdm(datanames)
        # fw = open(os.path.join(cfg.out_dir,'dataset.txt'), 'w') 
        for dataname in pbar:
            pbar.set_description('Model Processing ')
            if os.path.splitext(dataname)[1] == '.npy':
                res=64
                data = np.load(os.path.join(input_dir,dataname))
                voxel = torch.from_numpy(data.astype(np.int32)).to(device).reshape(res,res,res)
                hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
                all_node=voxel.clone()
                base_coo=torch.nonzero(voxel, as_tuple=False)
                for i in hex:
                    coo=(base_coo+i)%res
                    all_node[coo[:, 0], coo[:, 1], coo [:, 2]] = 1
                all_node=torch.nonzero(all_node, as_tuple=False)
                DOF[0]+=all_node.shape[0]
                res=128
                voxel=voxel.repeat(2, 2, 2)
                hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
                all_node=voxel.clone()
                base_coo=torch.nonzero(voxel, as_tuple=False)
                for i in hex:
                    coo=(base_coo+i)%res
                    all_node[coo[:, 0], coo[:, 1], coo [:, 2]] = 1
                all_node=torch.nonzero(all_node, as_tuple=False)
                DOF[1]+=all_node.shape[0]

                res=256
                voxel=voxel.repeat(2, 2, 2)
                hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
                all_node=voxel.clone()
                base_coo=torch.nonzero(voxel, as_tuple=False)
                for i in hex:
                    coo=(base_coo+i)%res
                    all_node[coo[:, 0], coo[:, 1], coo [:, 2]] = 1
                all_node=torch.nonzero(all_node, as_tuple=False)
                DOF[2]+=all_node.shape[0]

                voxel_nbr+=1
        DOF=DOF*1.0/length*3
        print(input_dir)
        print(DOF)

        # print('Data Size: ',voxel_nbr)
    print('Done!')