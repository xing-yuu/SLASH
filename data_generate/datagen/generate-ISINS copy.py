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
device = torch.device('cpu')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    

    for i in range (0,len(cfg.voxel.input_dir)):
        voxel_nbr=0
        input_dir=cfg.voxel.input_dir[i]
        output_dir=cfg.out_dir[i]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        datanames = os.listdir(input_dir)
        # datanames = random.sample(datanames,1000)
        length=len(datanames)
        tips=0
        CH_res=np.zeros((length,6,6),dtype=np.float32)
        pbar = tqdm(datanames)
        for dataname in pbar:
            pbar.set_description('Model Processing ')
            if os.path.splitext(dataname)[1] == '.csv':
                for i in range(1):
                    E=1
                    E_soft=E*1e-6
                    v=0.3
                    # v=random.uniform(0.2,0.4)
                    C_H=isotropic_elastic_tensor(E,v).to(device)
                    C_H_soft=isotropic_elastic_tensor(E_soft,v).to(device)
                    X0,Ke,Fe=homo_helper.voxel_XKF(res,res,res,C_H)
                    X0_soft,Ke_soft,Fe_soft=homo_helper.voxel_XKF(res,res,res,C_H_soft)
                    K_node,F_node=assembly_node_matrix(Ke,Fe)
                    K_node_soft,F_node_soft=assembly_node_matrix(Ke_soft,Fe_soft)
                    # data = np.load(os.path.join(input_dir,dataname))
                    data = np.loadtxt(open(os.path.join(input_dir,dataname),"rb"),dtype=np.int32)
                    voxel = torch.from_numpy(data.astype(np.int32)).to(device).reshape(res,res,res)
                    
                    # get soft voxel, hard voxel, node
                    all_voxel=voxel.clone()
                    base_coo=torch.nonzero(voxel, as_tuple=False)
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            for k in [-1, 0, 1]:
                                coo=(base_coo+torch.tensor([i,j,k],device=base_coo.device)+res)%res
                                all_voxel[coo[:,0],coo[:,1],coo[:,2]]=1
                    soft_voxel=all_voxel-voxel
                    
                    base_coo=torch.nonzero(all_voxel, as_tuple=False)
                    node_type_hard=get_node_type(voxel)
                    node_type_soft=get_node_type(soft_voxel)

                    
                    node_type=node_type_soft*1000+node_type_hard
                    all_node=all_voxel.clone()
                    hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
                    for i in hex:
                        coo=(base_coo+i)%res
                        all_node[coo[:, 0], coo[:, 1], coo [:, 2]] = 1
                    all_node=torch.nonzero(all_node, as_tuple=False)
                    node_type=node_type[all_node[:, 0], all_node[:, 1], all_node [:, 2]]
                    np.savez(os.path.join(output_dir,str(("%05d"%voxel_nbr)))+'.npz',
                                coordinates=np.float32(all_node.cpu().numpy()),
                                node_type=np.float32(node_type.cpu().numpy()),
                                Ke=np.float32(Ke.cpu().numpy()),
                                Fe=np.float32(Fe.cpu().numpy()),
                                X0=np.float32(X0.cpu().numpy()),
                                Ke_soft=np.float32(Ke_soft.cpu().numpy()),
                                Fe_soft=np.float32(Fe_soft.cpu().numpy()),
                                X0_soft=np.float32(X0_soft.cpu().numpy()),
                                K_node_soft=np.float32(K_node_soft.cpu().numpy()),
                                F_node_soft=np.float32(F_node_soft.cpu().numpy()),
                                K_node=np.float32(K_node.cpu().numpy()),
                                F_node=np.float32(F_node.cpu().numpy()),
                                C_H=np.float32(C_H.cpu().numpy()),
                                C_H_soft=np.float32(C_H_soft.cpu().numpy()),
                                E=E,
                                E_soft=E_soft,
                                v=v,
                            )
                    
                    # print('Finish ',voxel_nbr)
                    # fw.write(str(("%05d"%voxel_nbr))+'.npz\n')  

                    voxel_nbr+=1
 
    print('Data Size: ',voxel_nbr)
    print('Done!')
    









