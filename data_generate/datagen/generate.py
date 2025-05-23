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
from homogenization_helper import *
from numerical_homogenization import *
from node import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def plot_distribution(data,name):
    print(data.shape)
    print(data)
    print(np.min(data), np.max(data))
    bins = np.arange(np.min(data), np.max(data), 0.002)
    hist, _ = np.histogram(data, bins=bins)
    plt.bar(bins[:-1], hist, width=0.002)  # 调整柱状图宽度为0.4以适应步长0.5
    plt.xlabel('range')
    plt.ylabel('data number')
    plt.title('distribution_'+name)
    plt.savefig(name)
    plt.cla()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__=='__main__': 
    '''
    
    Args:
            Ke: [shape: 24*24], Element stiffness matrix 
            Fe: [shape: 24*6]: Element load

        Returns:
            K_node: [shape: 256*3*81], Node stiffness matrix 
            F_node: [shape: 256*3*6], Node load

    '''
    isotropic_elastic_tensor(1, 0.35)
    setup_seed(100)
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

    
    # material 
    E=float(cfg.material.youngs_modulus)
   
    v=float(cfg.material.poisson_ratio)
    ##voxel
    cache_file='datagen/cache/fem_cache.npz'
    homo_helper=homogenization_helper(cache_file,device)

        
    
    res=cfg.voxel.resolution
    
    
    numerical_solver=numerical_homogenization(res,res,res,1,1,1,device)
    
    
    # exit()
    
    
    for i in range (0,len(cfg.voxel.input_dir)):
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
                E=1
                E_soft=E*1e-6
                v=0.3
                C_H=isotropic_elastic_tensor(E,v).to(device)
                X0,Ke,Fe=homo_helper.voxel_XKF(res,res,res,C_H)
                data = np.loadtxt(open(os.path.join(input_dir,dataname),"rb"),dtype=np.int32)
                # data = np.load(os.path.join(input_dir,dataname))
                # data = np.loadtxt(open(os.path.join(cfg.voxel.input_dir,dataname),"rb"),dtype=np.int32)
                voxel = torch.from_numpy(data.astype(np.int32)).to(device).reshape(res,res,res)
                # U=numerical_solver.solve_by_torch(voxel,Ke,Fe)
                # CH_GT=numerical_solver.homogenized(voxel,U,Ke,X0)
                
                np.savez(os.path.join(output_dir,str(("%06d"%voxel_nbr)))+'.npz',
                            CH_GT=np.float32(CH_GT.cpu().numpy()),
                            voxel=np.float32(voxel.cpu().numpy()).astype(bool),
                        )
                
                # print('Finish ',voxel_nbr)
                # fw.write(str(("%06d"%voxel_nbr))+'.npz\n')  

                voxel_nbr+=1

        # print('Data Size: ',voxel_nbr)
    print('Done!')
    









