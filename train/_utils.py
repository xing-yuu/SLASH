import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
import time
import copy
interpolation = ME.MinkowskiInterpolation()

def smooth_kernel():
    one_kernel=torch.eye(3, 3, dtype=torch.float)
    kernel=torch.zeros(27, 3, 3*27, dtype=torch.float)
    for i in range(27):
        kernel[i,:,i*3:(i+1)*3]=one_kernel
    return kernel
  

@torch.no_grad()
def feature_assembly_M(nodetype,coo,material,device):
    """Assemble node material information for network input, input features include Poisson's ratio, Young's modulus, and one-hot of node type.
        Args:
            nodetype (torch.Tensor), shape(b*N,):
                Node type of each node.
            material (dict):
                Material information.
            device
        Returns:
            feature (torch.Tensor), shape(b*N,5):
                feature[:,0]: Young's modulus.
                feature[:,1]: Poisson's ratio.
                feature[:,2]==1: It's surrounded by solid voxels.
                feature[:,3]==1: It's surrounded by soft voxels.
                feature[:,4]==1: There are both solid voxels and soft voxels around it.
                feature[:,5]==1: ANCHOR! 
    """

    CH=torch.from_numpy(material["C_H"]).to(device).reshape(-1)
    feature = CH.repeat(nodetype.shape[0], 1).reshape(-1,36)
    feature = torch.cat((feature,coo[:,1:]),dim=1)
    return feature

@torch.no_grad()
def coo_downsample(fine_coo,rate):
    """Calculate the legal nodes of each layer.
        Args:
            fine_coo (torch.Tensor), shape(b*N,4):
                The coordinates of each point in the smallest layer. [batch_ID,x,y,z]
            level (Int):
                The total number of times it has been shrunk.
        Returns:
            coo_list (List), shape(level,(b*N,4)):
                The legal nodes of each layer.
    """
    
    coarse_coo=fine_coo
    coarse_coo[:,1:4]=coarse_coo[:,1:4] // (2**rate)
    coarse_coo=torch.unique(coarse_coo, dim=0)
    return coarse_coo


def resample(sptensor,coo,rate=2):
    # rate coo比sptensor.C的比例，如果是>0则是下采样，小于0是上采样
    coo_=coo.clone()
    coo_[:,1:]*=rate
    fea=interpolation(sptensor,coo_)
    return fea
    return ME.SparseTensor(
        coordinates=coo.float(),
        features=fea,
    )
def del_redundant_node(all_node,valid_coo):
    coo=valid_coo.clone()
    coo[:,1:]/=(2**level)
    F=all_node.features_at_coordinates(coo.float())
    return ME.SparseTensor(
            coordinates=valid_coo.float(),
            features=F 
    )

def Gauss_K_mask(coo):
    Gauss_mask=[]
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                mask=((coo[:,1]%2==i) & (coo[:,2]%2==j) & (coo[:,3]%2==k))
                mask = torch.nonzero(mask,as_tuple=False).reshape(-1)
                Gauss_mask.append(mask)
    return Gauss_mask
def boundary_padding(coo,fea,r):

    coo_mapping={}
    coo_ori=[]
    coo_tar=[]
    coo_mask=torch.arange(0,coo.shape[0],device=coo.device)
    Gauss_mask=Gauss_K_mask(coo)
    # for i in [0,1]:
    #     for j in [0,1]:
    #         for k in [0,1]:
    #             mask=((coo[:,1]%2==i) & (coo[:,2]%2==j) & (coo[:,3]%2==k))
    #             mask = torch.nonzero(mask,as_tuple=False).reshape(-1)
    #             Gauss_mask.append(mask)
    coo_mapping['K_mask']=copy.deepcopy(Gauss_mask)
    # print('coo',coo)
    # print('K_mask',coo_mapping['K_mask'])
    # exit()
    # for i in range (8):
    #     print(torch.max(coo_mapping['K_mask'][i]),torch.min(coo_mapping['K_mask'][i]))
    for i in range(1,4):
        mask_0 = torch.nonzero(coo[:,i] == 0,as_tuple=False).reshape(-1)
        
        coo_0=coo[mask_0]
        fea_0=fea[mask_0]
        coo_0[:,i]=r
        
        mask_r = torch.nonzero(coo[:,i] == r-1,as_tuple=False).reshape(-1)
        coo_r=coo[mask_r]
        fea_r=fea[mask_r]
        coo_r[:,i]=-1
        
        coo_tar.append(torch.arange(coo.shape[0], coo.shape[0]+coo_0.shape[0]+coo_r.shape[0], device=coo.device))
        coo_ori.append(torch.cat((mask_0,mask_r), dim=0))
        coo = torch.cat((coo,coo_0,coo_r), dim=0)
        fea = torch.cat((fea,fea_0,fea_r), dim=0)
    # for i in range (0,3):
    #     for j in range (0,10):
    #         print('ii',coo_ori[i][j],coo[coo_ori[i][j]],coo_tar[i][j],coo[coo_tar[i][j]],(coo[coo_tar[i][j]]+3)%3)
    # exit()
    sorted_index = coo[:, 0].argsort()
    coo=coo[sorted_index]
    fea=fea[sorted_index]
    ids_restore = torch.argsort(sorted_index)
 
    for i in range (3):
        coo_ori[i]=ids_restore[coo_ori[i]]
        coo_tar[i]=ids_restore[coo_tar[i]]
    for i in range (8):
        Gauss_mask[i]=ids_restore[Gauss_mask[i]]
    coo_mask=ids_restore[coo_mask]
    coo_mapping['coo_ori']=coo_ori
    coo_mapping['coo_tar']=coo_tar
    coo_mapping['ori_mask']=coo_mask
    coo_mapping['Gauss_mask']=Gauss_mask
    # for i in range (coo.shape[0]):
    #     print('coo',i,coo[i])
    # # print('coo-2',coo)
    # for i in range (0,3):
    #     for j in range (0,10):
    #         print('ii',coo_mapping['coo_ori'][i][j],coo[coo_mapping['coo_ori'][i][j]],coo_mapping['coo_tar'][i][j],coo[coo_mapping['coo_tar'][i][j]],(coo[coo_mapping['coo_tar'][i][j]]+3)%3)
    # exit()
    # for i in range (8):
    #     print(torch.max(coo_mapping['K_mask'][i]),torch.min(coo_mapping['K_mask'][i]))
    # print(torch.max(coo_mapping['K_mask']),torch.min(coo_mapping['K_mask']))
    return coo,fea,coo_mapping

# def reset_padding(U,coo_mapping):
#     for i in range(0,3):
#         U.F[:][coo_mapping['coo_tar'][i]]=U.F[coo_mapping['coo_ori'][i]].clone()
#     return U
def reset_padding(U,coo_mapping):

    Fea=U.F.clone()
    for i in range(0,3):
        Fea[coo_mapping['coo_tar'][i]]=U.F[coo_mapping['coo_ori'][i]].clone()

    return ME.SparseTensor(
            features=Fea,
            coordinate_map_key=U.coordinate_map_key, 
            coordinate_manager=U.coordinate_manager
    )


def up_resampling(sptensor,coo,coo_manage):
    coordinate_map_key,coordinate_manager=coo_manage
    coo_=coo.clone().float()
    coo_[:,1:]/=2

    # print(torch.max(coo_),torch.min(coo_),torch.max(sptensor.C),torch.min(sptensor.C),sptensor.C.shape,coo_.shape)
    fea=ME.MinkowskiInterpolation()(sptensor,coo_)
    
    return ME.SparseTensor(
            features=fea,
            coordinate_map_key=coordinate_map_key, 
            coordinate_manager=coordinate_manager
    )
# @torch.no_grad()
def smooth(K,f,U,r,K_mask,omega=1.):
    for i in range (8):
        base_coo=U.C[K_mask[i]]
        adjacent_node_coo=torch.zeros(27,base_coo.shape[0],base_coo.shape[1],dtype=torch.float32,device=base_coo.device)
        tips=0
        # Querying the 27 nodes surrounding each node, with the 13th node being itself.
        for xi in [-1, 0, 1]:
            for yj in [-1, 0, 1]:
                for zk in [-1, 0, 1]:
                    adjacent_node_coo[tips]=base_coo.clone()
                    adjacent_node_coo[tips,:,1:]=(adjacent_node_coo[tips,:,1:]+(torch.tensor([xi,yj,zk],dtype=torch.float32,device=base_coo.device))+r) % r
                    tips+=1

        adjacent_node_coo=adjacent_node_coo.reshape(-1,4)   
        for ax in range (3): 
            valid_u=U.features_at_coordinates(adjacent_node_coo).view(27,-1,6,3).transpose(0,1).transpose(2,3)
            valid_k=K.F[K_mask[i]].reshape(-1,27,3,3)[:,:,ax,:]
            valid_u=torch.einsum('pnj,pnjs->ps',valid_k,valid_u)
            valid_u=f.F[K_mask[i]].reshape(-1,3,6)[:,ax,:]-valid_u
            valid_u=torch.div(valid_u,valid_k[:,13,ax].view(-1, 1))
            v2=U.F.clone().reshape(-1,6,3)
            # print(v2.shape,K_mask[i].shape,valid_u.shape)
            v2[K_mask[i],:,ax]+=valid_u*omega
            U=ME.SparseTensor(
                features=v2.reshape(-1,18),
                coordinate_manager=U.coordinate_manager, 
                coordinate_map_key=U.coordinate_map_key  
            )
            # U.F[:][K_mask[i]]=v2.reshape(-1,18)
    U=ME.SparseTensor(
        features=U.F-torch.mean(U.F,dim=0),
        coordinate_manager=U.coordinate_manager, 
        coordinate_map_key=U.coordinate_map_key  
    )
    # U_mean=torch.mean(U.F,dim=0)
    return U
def smooth_jaobin(K,f,U,r,Lambda=2):
    base_coo=U.C
    adjacent_node_coo=torch.zeros(27,base_coo.shape[0],base_coo.shape[1],dtype=torch.float32,device=base_coo.device)
    tips=0
    # Querying the 27 nodes surrounding each node, with the 13th node being itself.
    for xi in [-1, 0, 1]:
        for yj in [-1, 0, 1]:
            for zk in [-1, 0, 1]:
                adjacent_node_coo[tips]=base_coo.clone()
                adjacent_node_coo[tips,:,1:]=(adjacent_node_coo[tips,:,1:]+(torch.tensor([xi,yj,zk],dtype=torch.float32,device=base_coo.device))+r) % r
                tips+=1

    adjacent_node_coo=adjacent_node_coo.reshape(-1,4)   
    valid_u=U.features_at_coordinates(adjacent_node_coo).view(27,-1,6,3).transpose(0,1).transpose(2,3)
    valid_k=K.F.reshape(-1,27,3,3).clone()
    valid_k[:, :, :, :] = valid_k[:, :, :, :] +  (torch.eye(3,device =valid_k.device)*Lambda).reshape(1,1,3,3)

    valid_u=torch.einsum('pnij,pnjs->pis',valid_k,valid_u)
    valid_u=f.F.reshape(-1,3,6)-valid_u
    K_diag=torch.diagonal(valid_k[:,13,:,:], offset=0,dim1=-2,dim2=-1)

    valid_u=torch.div(valid_u,(K_diag.unsqueeze(-1)))
    # print(torch.max(valid_u),torch.min(valid_u))
    U=ME.SparseTensor(
        features=valid_u.transpose(1,2).reshape(-1,18)+U.F,
        coordinate_manager=U.coordinate_manager, 
        coordinate_map_key=U.coordinate_map_key  
    )
    return U

# @torch.no_grad()
def bicgstab_(K, f, U,adjacent_node_coo , max_iter=30):
    device=adjacent_node_coo.device
    fixed_ID = torch.cumsum(torch.bincount(U.C[:,0]),dim=0)
    fixed_ID = torch.cat((torch.tensor([0],device=device), fixed_ID))
    fixed_U=U.F[fixed_ID[:-1],:]
    
    valid_k=K.F.view(-1,27,3,3).contiguous()
    # ci=
    time_start = time.perf_counter()  
    valid_u=torch.einsum('pnij,pnjs->pis',valid_k,U.features_at_coordinates(adjacent_node_coo).view(27,-1,6,3).permute(1,0,3,2))
    # .reshape(-1,18)
    time_end = time.perf_counter()  

    r=f.F.reshape(-1,3,6)-valid_u
    r_hat=r.clone()
    
    rho_old  = torch.ones(U.F.shape[0],6,device=device)
    rho_new = torch.zeros_like(rho_old,device=device)
    alpha = torch.ones(U.F.shape[0],6,device=device)
    omega = torch.ones(U.F.shape[0],6,device=device)

    # rho_old  = torch.ones(6,device=device)
    # rho_new = torch.zeros(6,device=device)
    # alpha = torch.ones(6,device=device)
    # omega = torch.ones(6,device=device)

    v = torch.zeros_like(f.F,device=device)
    p = torch.zeros_like(f.F,device=device)

    x = U.F.clone().view(-1,6,3).transpose(1,2)
    
    for i in range(max_iter):
        # alpha = alpha.clone()
        # omega = omega.clone()
        # rho_new = rho_new.clone()

        for k in range (0,fixed_ID.shape[0]-1):
            rho_new[fixed_ID[k]:fixed_ID[k+1],:] = torch.einsum('pij,pij->j', r_hat.view(-1,3,6)[fixed_ID[k]:fixed_ID[k+1],:,:].clone(), r.view(-1,3,6)[fixed_ID[k]:fixed_ID[k+1],:,:].clone()).repeat(fixed_ID[k+1]-fixed_ID[k], 1)
        # rho_new=torch.sum(rho_new,dim=0)
        
        beta = (torch.div(rho_new.clone() , rho_old.clone()) * (alpha.clone() / omega.clone()))
        # print(beta.shape)
        p = (r.view(-1,3,6).clone() + torch.einsum('qu,qvu->qvu',beta.clone() , (p.view(-1,3,6).clone() - torch.einsum('pj,pij->pij',omega.clone() ,v.view(-1,3,6).clone())))).clone()
        # p = (r.view(-1,3,6) + torch.einsum('u,qvu->qvu',beta , (p.view(-1,3,6) - torch.einsum('j,pij->pij',omega ,v.view(-1,3,6))))).clone()
        
        p_sp=ME.SparseTensor(
            features=p.view(-1,18).clone(),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )
        
        # time_start = time.perf_counter()  
        v =(torch.einsum('pnab,pnbs->pas',valid_k,p_sp.features_at_coordinates(adjacent_node_coo).view(27,-1,3,6).transpose(0,1))).clone()

        # alpha = torch.div(rho_new , torch.einsum('pij,pij->j', r_hat.view(-1,3,6), v))
        for k in range (0,fixed_ID.shape[0]-1):
            alpha[fixed_ID[k]:fixed_ID[k+1],:] = torch.div(rho_new[fixed_ID[k]:fixed_ID[k+1],:].clone(), torch.einsum('pij,pij->j', r_hat.view(-1,3,6).clone()[fixed_ID[k]:fixed_ID[k+1],:,:], v[fixed_ID[k]:fixed_ID[k+1],:,:].clone())).clone()

        s = (r.view(-1,3,6).clone() - torch.einsum('pj,pij->pij',alpha.clone() ,v.view(-1,3,6).clone())).clone()
        # s = (r.view(-1,3,6) - torch.einsum('j,pij->pij',alpha ,v.view(-1,3,6))).clone()
        
        s_sp=ME.SparseTensor(
            features=s.view(-1,18).clone(),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )

        t = torch.einsum('pnij,pnjs->pis',valid_k,s_sp.features_at_coordinates(adjacent_node_coo).view(27,-1,3,6).transpose(0,1)).clone()
        
        # torch.matmul(A, s)
        # omega = torch.div(torch.einsum('pij,pij->j', t, s) , torch.einsum('pij,pij->j', t, t))
        for k in range (0,fixed_ID.shape[0]-1):
            omega[fixed_ID[k]:fixed_ID[k+1],:] = torch.div(torch.einsum('pij,pij->j', t.clone()[fixed_ID[k]:fixed_ID[k+1]], s.clone()[fixed_ID[k]:fixed_ID[k+1]]) , torch.einsum('pij,pij->j', t.clone()[fixed_ID[k]:fixed_ID[k+1]], t.clone()[fixed_ID[k]:fixed_ID[k+1]])).repeat(fixed_ID[k+1]-fixed_ID[k], 1)
        # omega = torch.div(torch.einsum('pij,pij->j', t, s) , torch.einsum('pij,pij->j', t, t))

        # np.dot(t, t)
        # x = torch.add(x , torch.einsum('j,pij->pij',alpha ,p.view(-1,3,6)) +torch.einsum('j,pij->pij',omega ,s.view(-1,3,6)) )
        x = torch.add(x.clone() , torch.einsum('pj,pij->pij',alpha.clone() ,p.clone().view(-1,3,6)) +torch.einsum('pj,pij->pij',omega.clone() ,s.clone().view(-1,3,6)) ).clone()

        r = (s -torch.einsum('pj,pij->pij',omega.clone() ,t.clone().view(-1,3,6))).clone()
        # r = (s -torch.einsum('j,pij->pij',omega ,t.view(-1,3,6))).clone()
        rho_old = rho_new.clone()

        x[fixed_ID[:-1],:,:]=fixed_U.view(-1,18).reshape(-1,6,3).transpose(1,2)
        
    return ME.SparseTensor(
            features=x.transpose(1,2).view(-1,18).clone(),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )

def CG(K, f, U,adjacent_node_coo ,r, max_iter=3):
    fixed_ID = torch.cumsum(torch.bincount(U.C[:,0]),dim=0)
    fixed_ID[-1]=0

    fixed_U=U.F[fixed_ID]
    
    valid_k=K.F.view(-1,27,3,3).contiguous()

    valid_u=torch.einsum('pnij,pnjs->pis',valid_k,U.features_at_coordinates(adjacent_node_coo).view(27,-1,6,3).permute(1,0,3,2))


    r=f.F.reshape(-1,3,6)-valid_u
    # r_hat=r.clone()
    rho_old = torch.einsum('pij,pij->j', r.view(-1,3,6), r.view(-1,3,6))
    
    x = U.F.clone().view(-1,6,3).transpose(1,2)
    p = r.clone()
    for i in range(max_iter):
        
        p_sp=ME.SparseTensor(
            features=p.view(-1,18),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )
        
        v =torch.einsum('pnab,pnbs->pas',valid_k,p_sp.features_at_coordinates(adjacent_node_coo).view(27,-1,3,6).transpose(0,1))
        alpha = torch.div(rho_old , torch.einsum('pij,pij->j', p.view(-1,3,6), v))

        x += alpha * p
        r -= alpha * v
        rho_new = torch.einsum('pij,pij->j', r.view(-1,3,6), r.view(-1,3,6))

        p = r + (rho_new/rho_old) * p
        rho_old = rho_new
    
    return ME.SparseTensor(
            features=x.transpose(1,2).view(-1,18),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )
def CG_2(K, f, U,adjacent_node_coo ,r, max_iter=70):
    device=adjacent_node_coo.device
    fixed_ID = torch.cumsum(torch.bincount(U.C[:,0]),dim=0)
    fixed_ID = torch.cat((torch.tensor([0],device=device), fixed_ID))
    fixed_U=U.F[fixed_ID[:-1],:]
    # print(fixed_ID)
    # exit()
    
    valid_k=K.F.view(-1,27,3,3).contiguous()
    # ci=
    time_start = time.perf_counter()  
    valid_u=torch.einsum('pnij,pnjs->pis',valid_k,U.features_at_coordinates(adjacent_node_coo).view(27,-1,6,3).permute(1,0,3,2))
    # .reshape(-1,18)
    time_end = time.perf_counter()  

    r=f.F.reshape(-1,3,6)-valid_u
    K_diag=torch.diagonal(valid_k[:,13,:,:], offset=0,dim1=-2,dim2=-1)


    K_inv = torch.div(1.0,K_diag)

    # print(K_inv*K_diag)
    # exit()
    X = U.F.clone().view(-1,6,3).transpose(1,2)
    p = r.clone()
    rs_old=torch.zeros(U.F.shape[0],6,device=device)
    rs_new=torch.zeros(U.F.shape[0],6,device=device)
    alpha = torch.ones(U.F.shape[0],6,device=device)
    for k in range (0,fixed_ID.shape[0]-1):
        rs_old[fixed_ID[k]:fixed_ID[k+1],:] = torch.einsum('pij,pij->j',r[fixed_ID[k]:fixed_ID[k+1],:,:] ,r[fixed_ID[k]:fixed_ID[k+1],:,:])

    for i in range(max_iter):
        p_sp=ME.SparseTensor(
            features=p.view(-1,18).clone(),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )
        
        AP =(torch.einsum('pnab,pnbs->pas',valid_k,p_sp.features_at_coordinates(adjacent_node_coo).view(27,-1,3,6).transpose(0,1))).clone()


        for k in range (0,fixed_ID.shape[0]-1):
            alpha[fixed_ID[k]:fixed_ID[k+1],:] = torch.div(rs_old[fixed_ID[k]:fixed_ID[k+1],:].clone(), torch.einsum('pij,pij->j', p.view(-1,3,6).clone()[fixed_ID[k]:fixed_ID[k+1],:,:], AP[fixed_ID[k]:fixed_ID[k+1],:,:].clone())).clone()

        X = X + torch.einsum('pj,pij->pij',alpha, p)
        # X[fixed_ID[:-1],:,:]=fixed_U.view(-1,18).reshape(-1,6,3).transpose(1,2)
        
        r = r - torch.einsum('pj,pij->pij',alpha, AP)
        

        for k in range (0,fixed_ID.shape[0]-1):
            rs_new[fixed_ID[k]:fixed_ID[k+1],:] = torch.einsum('pij,pij->j',r[fixed_ID[k]:fixed_ID[k+1],:,:] ,r[fixed_ID[k]:fixed_ID[k+1],:,:])

        p = r + torch.einsum('pj,pij->pij',torch.div(rs_new , rs_old) ,p)
        rs_old = rs_new.clone()
        

    return ME.SparseTensor(
            features=X.transpose(1,2).view(-1,18),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )
    
    
