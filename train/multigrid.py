import torch
import math
import MinkowskiEngine as ME
import torch.nn.functional as F
import time



@torch.no_grad()
def node_KF_assembly(batch_id,nodetype,K_node,F_node):
    """Assemble global node stiffness matrix and global node load according to node type, for loss function.
        Args:
            nodetype (torch.Tensor), shape(b*N,):
                Node type of each node.
            material (Dict):
                Material information.
            device
        Returns:
            features_k (torch.Tensor), shape(b*N,27*9):
                Stiffness matrix for each node
            features_f (torch.Tensor), shape(b*N,6*3):
                Load matrix for each node
    """
    # print(torch.max(batch_id),K_node.shape,torch.max(nodetype),torch.min(nodetype))
    features_k=K_node["hard"][batch_id.long(),nodetype.long()%1000,:]
    features_f=F_node["hard"][batch_id.long(),nodetype.long()%1000,:]
    features_k=features_k+K_node["soft"][batch_id.long(),nodetype.long()//1000,:]
    features_f=features_f+F_node["soft"][batch_id.long(),nodetype.long()//1000,:]
    return features_k,features_f

@torch.no_grad()
def node_KF_assembly_hard(batch_id,nodetype,K_node,F_node):
    """Assemble global node stiffness matrix and global node load according to node type, for loss function.
        Args:
            nodetype (torch.Tensor), shape(b*N,):
                Node type of each node.
            material (Dict):
                Material information.
            device
        Returns:
            features_k (torch.Tensor), shape(b*N,27*9):
                Stiffness matrix for each node
            features_f (torch.Tensor), shape(b*N,6*3):
                Load matrix for each node
    """
    # print(torch.max(batch_id),K_node.shape,torch.max(nodetype),torch.min(nodetype))
    features_k=K_node["hard"][batch_id.long(),nodetype.long()%1000,:]
    features_f=F_node["hard"][batch_id.long(),nodetype.long()%1000,:]
    features_k=features_k+K_node["soft"][batch_id.long(),nodetype.long()//1000,:]
    features_f=features_f+F_node["soft"][batch_id.long(),nodetype.long()//1000,:]
    return features_k,features_f

@torch.no_grad()
def feature_assembly_M(batch_id,nodetype,CH_hard,CH_soft):
    """Assemble node material information for network input, input features include Poisson's ratio, Young's modulus, and one-hot of node type.
        Args:
            nodetype (torch.Tensor), shape(b*N,):
                Node type of each node.
            material (dict):
                Material information.
            device
        Returns:
            feature (torch.Tensor), shape(b*N,5):

    """
    feature=torch.zeros(batch_id.shape[0],36,device=batch_id.device)
    feature[(nodetype.long()%1000)>0] = CH_hard[batch_id[(nodetype.long()%1000)>0].long(),:].reshape(-1,36)
    feature[(nodetype.long()//1000)>0] = CH_soft[batch_id[(nodetype.long()//1000)>0].long(),:].reshape(-1,36)
    return feature

@torch.no_grad()
def calculate_legal_coordinates(fine_coo,level):
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
    coo_list=[fine_coo]
    for i in range(1,level+1):
        coarse_coo=coo_list[i-1].clone()
        mask=torch.all(coarse_coo[:,1:]%2==0,dim=1)
        coarse_coo=coarse_coo[mask]
        coarse_coo[:,1:]=coarse_coo[:,1:]/2
        coo_list.append(coarse_coo)
    return coo_list
@torch.no_grad()
def calculate_adj_coo(coo_list,r):
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
    adj_coo_list=[]
    for base_coo in coo_list:
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
        # print(adjacent_node_coo.shape)
        adj_coo_list.append(adjacent_node_coo.float())
        r/=2
    # exit()
    return adj_coo_list

@torch.no_grad()
def calculate_level_K_list(node_stiffness,level_coo_list,batch_size,r,level):
    device=node_stiffness.F.device
    # print(device)
    level_K=[decomposed_K(node_stiffness,r)]
    dense_k,_,_=node_stiffness.dense(shape=torch.Size((batch_size,243, r, r,r)),min_coordinate=torch.IntTensor([0,0,0]))
    dense_k=dense_k.permute(0,2,3,4,1)

    # print('K0',torch.max(level_K[0].F),torch.min(level_K[0].F))
    for i in range (1, level):
        level_coo=level_coo_list[i].clone()
        coarse_r=r//(2**i)
        dense_k=fine2coarse_Knode(dense_k,level_coo,batch_size,coarse_r,device)
        fea_k=dense_k[level_coo[:,0],level_coo[:,1],level_coo[:,2],level_coo[:,3],:]
        level_K.append(decomposed_K(ME.SparseTensor(coordinates=level_coo_list[i].float(), features=fea_k),coarse_r))

    return level_K

@torch.no_grad()
def decomposed_K(node_stiffness,r):
    K=[]
    list_of_coords, list_of_featurs = node_stiffness.decomposed_coordinates_and_features
    for coo, fea in zip(list_of_coords, list_of_featurs):
        K.append(assembly_sp_K(coo, fea,r))
    return K
        
def fine2coarse_interpolation(res_h,coarse_coo,coarse_r):
    """Interpolate the features of the finer layer to the coarse layer.
        Args:
            res_h (MinkowskiEngine.SparseTensor), shape(b,C,fine_r,fine_r,fine_r):
                Features of the finer layer.
            coarse_coo (torch.Tensor), shape(b*N,4):
                Coordinates of the coarser layer.
            fine_r (Int):
                Resolution of the finer layer. 
            offset (Int):
                Coordinate offset, 1 or 0. 
            coordinate_manager (MinkowskiCoordinateManager.CoordinateManager)
                A universal coordinate manager of the coarser layer.
            device 
        Returns:
            res_2h (MinkowskiEngine.SparseTensor), shape(b,C,fine_r/2,fine_r/2,fine_r/2):
                Features of the coarser layer.     
    """
    device=res_h.F.device
    fine_r=coarse_r*2
    all_coo=torch.zeros(27,coarse_coo.shape[0],coarse_coo.shape[1],dtype=torch.float32,device=device)
    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    all_coo[:,:,1:] = (coarse_coo[:,1:].unsqueeze(0)*2 + offsets.view(27,1,3)+ fine_r) % fine_r
    all_coo[:,:,0]=coarse_coo[:,0]
    ratio=torch.div(1.,torch.pow(2,torch.sum(torch.abs(offsets),dim=1)))
    all_coo=all_coo.reshape(-1,4)
    return torch.einsum('pgi,p->gi',res_h.features_at_coordinates(all_coo).reshape(27,coarse_coo.shape[0],-1),ratio)


def calculate_residual_2D(coo,K,U,F,batch_size):
    res=[]
    for i in range (0,batch_size):
        mask=coo[:,0]==i
        u_=U[mask].reshape(-1,6,3).transpose(1,2).reshape(-1,6)
        f_=F[mask].reshape(-1,3,6).reshape(-1,6)
        R =  f_- torch.sparse.mm(K[i], u_)
        R=R.reshape(-1,3,6).reshape(-1,18)
        res.append(R)
    return torch.cat(res,dim=0)

def coarse2fine_interpolation_error(error_2h,fine_coo,coarse_r,device,coordinate_manager=None,coordinate_map_key=None):
    feature_channel=error_2h.F.shape[1]
    fine_r=coarse_r*2
    # error_h=torch.zeros(fine_coo.shape[0],18,device=device, requires_grad=True)
    coo_2h=torch.zeros(27,fine_coo.shape[0],fine_coo.shape[1],dtype=torch.float32,device=device)
    # mask=torch.zeros(27,fine_coo.shape[0],dtype=torch.float32,device=device)

    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    coo_2h[:,:,1:] = (fine_coo[:,1:].unsqueeze(0) + offsets.view(27,1,3)+ fine_r) % fine_r
    coo_2h[:,:,0]=fine_coo[:,0]

    ratio=torch.div(1.,torch.pow(2,torch.sum(torch.abs(offsets),dim=1)))
    mask=~((coo_2h[:,:,1:4] % 2).any(dim=-1))

    coo_2h[:,:,1:4]=coo_2h[:,:,1:4]//2


    coo_2h=coo_2h.reshape(-1,4)
    error_h_feature=error_2h.features_at_coordinates(coo_2h).reshape(27,-1,feature_channel).transpose(0,1)
    mask=mask.transpose(0,1)
    # print(error_h_feature.shape,mask.t().shape,ratio.shape)
    error_h_feature=torch.einsum('pnj,pn,n->pj',error_h_feature,mask,ratio)
    # *0.125
    if coordinate_manager is not None:
        return ME.SparseTensor(
            error_h_feature,
            coordinate_manager=coordinate_manager, 
            coordinate_map_key=coordinate_map_key)

    return error_h_feature

def calculate_residual_2D(coo,K_model_list,U,F,batch_size):
    res=[]
    for i in range (0,batch_size):
        mask=coo[:,0]==i
        u_=U[mask].reshape(-1,6,3).transpose(1,2).reshape(-1,6)
        f_=F[mask].reshape(-1,3,6).reshape(-1,6)
        R =  f_- torch.sparse.mm(K_model_list[i], u_)
        R=R.reshape(-1,3,6).reshape(-1,18)
        res.append(R)
    return torch.cat(res,dim=0)


def calculate_residual(K,U,F,r,device):
    """Calculate residual according to KU-F
        Args:
            K
                shape(b,27*3*3,r,r,r)
            U
                shape(b,6*3,r,r,r)
            F
                shape(b,3*6,r,r,r)

        Returns:
            R
                shape(b,3*6,r,r,r)
    """
    node_num=K.C.shape[0]
    base_coo=K.C

    coo_t=torch.zeros(27,K.C.shape[0],K.C.shape[1],dtype=torch.float32,device=device)

    K_feature=K.F.view(-1,27,3,3)

    coo_t=torch.zeros(27,base_coo.shape[0],base_coo.shape[1],dtype=torch.float32,device=device)

    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    coo_t[:,:,1:] = (base_coo[:,1:].unsqueeze(0) + offsets.view(27,1,3)+ r) % r
    coo_t[:,:,0] = base_coo[:,0]


    coo_t=coo_t.reshape(-1,4)
    # shape N,27,3,6
    U_feature=U.features_at_coordinates(coo_t).reshape(27,node_num,-1,3).permute(1,0,3,2)
    # shape N,27,3,6->shape N,(3,6)
    residual=torch.einsum('cnki,cnij->ckj',K_feature,U_feature).reshape(node_num,-1)
    # print('res',torch.max(F.C),torch.min(F.C),torch.max(U.C),torch.min(U.C),torch.max(K.C),torch.min(K.C))        
    R= ME.SparseTensor(
            coordinates=F.C,
            features=F.F-residual # must share the same coordinate manager
    )
    return R

@torch.no_grad()
def fine2coarse_Knode(Knode_h,coarse_coo,batch_size,coarse_r,device):
    """The Stiffness matrix is reduced 
        Args:
            Knode_h
                shape(b,27*3*3,r*2,r*2,r*2)
        Returns:
            Knode_2h
                shape(b,27*3*3,r,r,r)
    """
    
    fine_r=coarse_r*2
    Knode_h=Knode_h.reshape(batch_size,fine_r,fine_r,fine_r,3,3,3,9)
    
    B_h=torch.zeros(batch_size,coarse_r,coarse_r,coarse_r,5,5,5,3*3,dtype=Knode_h.dtype,device=device)
    Knode_2h=torch.zeros(batch_size,coarse_r,coarse_r,coarse_r,3,3,3,3*3,dtype=Knode_h.dtype,device=device)
    # print('denseK',coarse_r,B_h.shape)
    # print(coarse_coo)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                ratio=1.0/pow(2.0,(abs(i)+abs(j)+abs(k)))
                fine_coo=coarse_coo.clone()
                fine_coo[:,1:4]=((fine_coo[:,1:4]*2+torch.tensor([i,j,k],device=device)+fine_r)%fine_r)
                fine_coo=fine_coo.long()
                B_h[coarse_coo[:,0],coarse_coo[:,1],coarse_coo[:,2],coarse_coo[:,3],i+1:i+4,j+1:j+4,k+1:k+4,:]+=Knode_h[fine_coo[:,0],fine_coo[:,1],fine_coo[:,2],fine_coo[:,3],:]*ratio
    
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                for ib in [-1, 0, 1]:
                    for jb in [-1, 0, 1]:
                        for kb in [-1, 0, 1]:
                            coob=torch.tensor([i*2+ib,j*2+jb,k*2+kb],device=device)
                            ratio=1.0/pow(2.0,(abs(ib)+abs(jb)+abs(kb)))
                            if torch.any(abs(coob)>2):
                                continue
                            Knode_2h[coarse_coo[:,0],coarse_coo[:,1],coarse_coo[:,2],coarse_coo[:,3],i+1,j+1,k+1,:]+=B_h[coarse_coo[:,0],coarse_coo[:,1],coarse_coo[:,2],coarse_coo[:,3],coob[0]+2,coob[1]+2,coob[2]+2,:]*ratio
    Knode_2h=Knode_2h.reshape(batch_size,coarse_r,coarse_r,coarse_r,-1)
    # *0.125
    return Knode_2h


def displacement_regularization(U,batch_size):
    loss_U=torch.zeros(18,requires_grad=True,device=U.device)
    for i in range (batch_size):
        loss_U=loss_U+torch.mean(U.features_at(i),dim=0)**2
    return torch.mean(loss_U)


def smooth_pcg_train(K, f, U,fixed_ID,R, max_iter=70):
    device=U.device
    mask=torch.ones(U.shape[0],1,device=device)
    mask[fixed_ID[:-1],:]=0

    adjacent_node_coo = torch.zeros(27, U.shape[0], 4,dtype=torch.float32, device=device)
    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    adjacent_node_coo[:,:,1:] = (U.C[:,1:].unsqueeze(0) + offsets.view(27,1,3)+ R) % R
    adjacent_node_coo[:,:,0]=U.C[:,0]
    adjacent_node_coo=adjacent_node_coo.view(-1,4)
    # fixed_U=U.F[fixed_ID[:-1],:]
    valid_k=K.F.view(-1,27,3,3).contiguous()
    valid_u=torch.einsum('pnij,pnjs->pis',valid_k,U.features_at_coordinates(adjacent_node_coo).view(27,-1,6,3).permute(1,0,3,2))
    r=f.F.reshape(-1,3,6)-valid_u
    if torch.linalg.norm(r)<1e-8:
        return U

    K_diag=torch.diagonal(valid_k[:,13,:,:], offset=0,dim1=-2,dim2=-1)

    K_inv = torch.div(1.0,K_diag)
    # print('K——inv',torch.max(K_inv),torch.min(K_inv))
    Z = torch.einsum('pj,pjf->pjf',K_inv,r)
    X = U.F.clone().view(-1,6,3).transpose(1,2)
    # p = Z.clone()
    p = ME.SparseTensor(Z.view(-1,18),
                        coordinate_manager=U.coordinate_manager, 
                        coordinate_map_key=U.coordinate_map_key  
                        )
    rs_old=torch.zeros(U.F.shape[0],6,device=device)
    rs_new=torch.zeros(U.F.shape[0],6,device=device)
    alpha = torch.ones(U.F.shape[0],6,device=device)
    
    
    for k in range (0,fixed_ID.shape[0]-1):
        rs_old[fixed_ID[k]:fixed_ID[k+1],:] = torch.einsum('pij,pij->j',r[fixed_ID[k]:fixed_ID[k+1],:,:] ,Z[fixed_ID[k]:fixed_ID[k+1],:,:])


    for i in range(max_iter):
        
        AP =(torch.einsum('pnab,pnbs->pas',valid_k,p.features_at_coordinates(adjacent_node_coo).view(27,-1,3,6).transpose(0,1) ))
        # print('mm',time_end2-time_end,time_end-time_start)
        alpha_list=[]
        for k in range (0,fixed_ID.shape[0]-1):
            alpha[fixed_ID[k]:fixed_ID[k+1],:]=torch.div(rs_old[fixed_ID[k]:fixed_ID[k+1],:], torch.einsum('pij,pij->j', p.F.view(-1,3,6)[fixed_ID[k]:fixed_ID[k+1],:,:], AP[fixed_ID[k]:fixed_ID[k+1],:,:]))
        #     alpha_list.append( )
        # alpha=torch.cat(alpha_list,dim=0)
        
        X = X + torch.einsum('pj,pij->pij',alpha, p.F.view(-1,3,6))*mask.view(-1,1,1)
        
        r = r - torch.einsum('pj,pij->pij',alpha, AP)
        
        Z = torch.einsum('pj,pjf->pjf',K_inv, r)
        # print('ZZ',torch.min(Z))
        # rs_new_list=[]
        for k in range (0,fixed_ID.shape[0]-1):
            rs_new[fixed_ID[k]:fixed_ID[k+1],:]=torch.einsum('pij,pij->j',r[fixed_ID[k]:fixed_ID[k+1],:,:] ,Z[fixed_ID[k]:fixed_ID[k+1],:,:]).repeat(fixed_ID[k+1]-fixed_ID[k],1)
            # rs_new_list.append()
        # rs_new=torch.cat(rs_new_list,dim=0)

        # p = Z.clone() + torch.einsum('pj,pij->pij',torch.div(rs_new.clone() , rs_old.clone()) ,p.clone())
        p = ME.SparseTensor((Z + torch.einsum('pj,pij->pij',torch.div(rs_new, rs_old) ,p.F.view(-1,3,6))).reshape(-1,18),
                        coordinate_manager=U.coordinate_manager, 
                        coordinate_map_key=U.coordinate_map_key  
                        )
        rs_old = rs_new.clone()
    # return X.transpose(1,2).view(-1,18)
    return ME.SparseTensor(
            features=X.transpose(1,2).view(-1,18),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )

def smooth_pcg_infer(U, K, f, fixed_ID,R, max_iter=70,adjacent_node_coo=None):
    device=U.device
    mask=torch.ones(U.shape[0],1,device=device)
    mask[fixed_ID[:-1],:]=0
    if adjacent_node_coo is None:
        adjacent_node_coo = torch.zeros(27, U.shape[0], 4,dtype=torch.float32, device=device)
        offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
        adjacent_node_coo[:,:,1:] = (U.C[:,1:].unsqueeze(0) + offsets.view(27,1,3)+ R) % R
        adjacent_node_coo[:,:,0]=U.C[:,0]
        adjacent_node_coo=adjacent_node_coo.view(-1,4)
    valid_k=K.F.view(-1,27,3,3).contiguous()
    valid_u=torch.einsum('pnij,pnjs->pis',valid_k,U.features_at_coordinates(adjacent_node_coo).view(27,-1,6,3).permute(1,0,3,2))
    r=f.F.reshape(-1,3,6)-valid_u
    
    if torch.linalg.norm(r)<1e-8:
        return U

    K_diag=torch.diagonal(valid_k[:,13,:,:], offset=0,dim1=-2,dim2=-1)
    K_inv = torch.div(1.0,K_diag)

    # print('K——inv',torch.max(K_inv),torch.min(K_inv))
    Z = torch.einsum('pj,pjf->pjf',K_inv,r)
    X = U.F.clone().view(-1,6,3).transpose(1,2)
    # p = Z.clone()
    p = ME.SparseTensor(Z.view(-1,18),
                        coordinate_manager=U.coordinate_manager, 
                        coordinate_map_key=U.coordinate_map_key  
                        )
    rs_old=torch.zeros(U.F.shape[0],6,device=device)
    rs_new=torch.zeros(U.F.shape[0],6,device=device)
    alpha = torch.ones(U.F.shape[0],6,device=device)
    
    
    for k in range (0,fixed_ID.shape[0]-1):
        rs_old[fixed_ID[k]:fixed_ID[k+1],:] = torch.einsum('pij,pij->j',r[fixed_ID[k]:fixed_ID[k+1],:,:] ,Z[fixed_ID[k]:fixed_ID[k+1],:,:])


    for i in range(max_iter):
        
        AP = (torch.einsum('pnab,pnbs->pas',valid_k,p.features_at_coordinates(adjacent_node_coo).view(27,-1,3,6).transpose(0,1) ))
        # print('mm',time_end2-time_end,time_end-time_start)
        # alpha_list=[]
        for k in range (0,fixed_ID.shape[0]-1):
            # alpha_list.append( torch.div(rs_old[fixed_ID[k]:fixed_ID[k+1],:], torch.einsum('pij,pij->j', p.F.view(-1,3,6)[fixed_ID[k]:fixed_ID[k+1],:,:], AP[fixed_ID[k]:fixed_ID[k+1],:,:])))
            alpha[fixed_ID[k]:fixed_ID[k+1],:] = torch.div(rs_old[fixed_ID[k]:fixed_ID[k+1],:], torch.einsum('pij,pij->j', p.F.view(-1,3,6)[fixed_ID[k]:fixed_ID[k+1],:,:], AP[fixed_ID[k]:fixed_ID[k+1],:,:]))

        # alpha=torch.cat(alpha_list,dim=0)
        
        X.add_(torch.einsum('pj,pij->pij',alpha, p.F.view(-1,3,6))*mask.view(-1,1,1))
        
        r.sub_(torch.einsum('pj,pij->pij',alpha, AP))
        
        Z = torch.einsum('pj,pjf->pjf',K_inv, r)

        # rs_new_list=[]
        for k in range (0,fixed_ID.shape[0]-1):
            # rs_new_list.append()
            rs_new[fixed_ID[k]:fixed_ID[k+1],:]=torch.einsum('pij,pij->j',r[fixed_ID[k]:fixed_ID[k+1],:,:] ,Z[fixed_ID[k]:fixed_ID[k+1],:,:]).repeat(fixed_ID[k+1]-fixed_ID[k],1)

        # p = Z.clone() + torch.einsum('pj,pij->pij',torch.div(rs_new.clone() , rs_old.clone()) ,p.clone())
        p.F[:]=(Z + torch.einsum('pj,pij->pij',torch.div(rs_new, rs_old) ,p.F.view(-1,3,6))).reshape(-1,18)

        rs_old.copy_(rs_new)
    # return X.transpose(1,2).view(-1,18)
    return ME.SparseTensor(
            features=X.transpose(1,2).view(-1,18),
            coordinate_manager=U.coordinate_manager, 
            coordinate_map_key=U.coordinate_map_key  
        )


def assembly_sp(coo, K_value, F_value,r):
    """assemble stiffness matrix K and force matrix for solid cell

    Args:
        voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
        Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
        Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 

    Returns:
        K(torch.cuda.Soarse_COO_Tensor)
        F(torch.cuda.FloatTensor)
    """
    device=coo.device
    index=torch.zeros(r,r,r,dtype=torch.int32,device=device)
    node_index_0 = torch.arange(0, coo.shape[0],dtype=torch.int32,device=device)
    # print(node_index_0.dtype,coo.dtype,index.dtype)
    index[coo[:,0],coo[:,1],coo[:,2]]=node_index_0


    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    all_coo= (coo.unsqueeze(0) + offsets.view(27,1,3)+ r) % r
    all_coo=all_coo.reshape(27,-1,3).transpose(0,1).reshape(-1,3)
    node_index_1=index[all_coo[:,0],all_coo[:,1],all_coo[:,2]]*3
    node_index_1 = node_index_1.unsqueeze(1).repeat(1, 3) 
    # print(node_index_1.shape,node_index_1[:,1].shape)
    # print(node_index_1[0,0])
    node_index_1[:,1]+=1
    # print(node_index_1[0,0])
    node_index_1[:,2]+=2
    # print(node_index_1[0],node_index_1[1])
    # exit()
    node_index_0=torch.zeros(coo.shape[0],27,3,device=device)
    # print('yyy',torch.cuda.memory_allocated()/(1024**3))
    for t in [0,1,2]:
        node_index_0[:,:,t]=index[coo[:,0],coo[:,1],coo[:,2]].unsqueeze(1).repeat(1, 27)*3+t

    node_index_1=node_index_1.reshape(-1,1).repeat(1, 3).reshape(-1,3,3).transpose(-1,-2).reshape(1,-1)
    # print(node_index_1[0],node_index_1[1],node_index_1[2])
    # exit()
    node_index_0=node_index_0.reshape(-1,1).repeat(1, 3).reshape(1,-1)
    # node_index_0=node_index_0.to(torch.int16)
    # node_index_1=node_index_1.to(torch.int16)
    # print('ooo',torch.cuda.memory_allocated()/(1024**3))
    # .reshape(-1,1)
    # print(node_index_0[0],node_index_0[1],node_index_0[2])
    # node_index_0=node_index_0.reshape(-1,1)
    # print(K_value.shape)
    
    # del index
    # del all_coo
    K_value=K_value.reshape(-1)
    F_value=F_value.reshape(-1)
    # print(node_index_0.shape,node_index_1.shape,K_value.shape,3 * coo.shape[0],F_value.reshape(-1,3,6).view(-1,6).shape)
    K = torch.sparse_coo_tensor(torch.cat([node_index_0,node_index_1],dim=0), K_value, (3 * coo.shape[0], 3 * coo.shape[0]), device=device).coalesce()
    # .coalesce()
    F = F_value.reshape(-1,3,6).view(-1,6)
    # K = K.to(torch.float16)
    # F = F.to(torch.float16)

    return K, F


def smooth_PCG_sp_assembly(K,f,u0,maxit): 
    # K=K.coalesce()
    indices = K.indices()
    # rows, cols = indices[0], indices[1]
    values = K.values()
    n = K.size(0)
    
    # 提取对角线元素构造预条件子M_diag
    diag_mask = (indices[0] == indices[1])
    diag_rows = indices[0][diag_mask]
    diag_values = values[diag_mask]

    
    diag = torch.zeros(n, dtype=values.dtype, device=K.device)
    diag[diag_rows] = diag_values
    M_diag = 1.0 / diag
    u0=u0.reshape(-1,6,3).transpose(1,2).reshape(-1,6)
    r =  f- torch.sparse.mm(K, u0)
    z = M_diag.unsqueeze(1)*r
    p = z.clone()
    rho_old = torch.sum(r*z,dim=0)
    t1=time.time()
    for _ in range(maxit):
        Ap = torch.sparse.mm(K, p)
        alpha = torch.div(rho_old , torch.sum(p* Ap,dim=0)) 
        u0 = u0 + alpha * p
        r = r - alpha * Ap
        z_new = M_diag.unsqueeze(1) * r
        rho_new = torch.sum(r*z_new,dim=0)
        beta = rho_new / (rho_old )
        p = z_new + beta * p
        rho_old = rho_new
    t2=time.time()

    u0 = u0.contiguous().reshape(-1, 3 , 6).transpose(1, 2).reshape(-1,18)
    
    u0=u0-torch.mean(u0,dim=0)
    return u0
def assembly_sp_K(coo, K_value,r):
    coo=coo.long()
    device=coo.device
    index=torch.zeros(r,r,r,dtype=torch.int32,device=device)
    node_index_0 = torch.arange(0, coo.shape[0],dtype=torch.int32,device=device)
    # print(node_index_0.dtype,coo.dtype,index.dtype)
    index[coo[:,0],coo[:,1],coo[:,2]]=node_index_0


    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    all_coo= (coo.unsqueeze(0) + offsets.view(27,1,3)+ r) % r
    all_coo=all_coo.reshape(27,-1,3).transpose(0,1).reshape(-1,3)
    node_index_1=index[all_coo[:,0],all_coo[:,1],all_coo[:,2]]*3
    node_index_1 = node_index_1.unsqueeze(1).repeat(1, 3) 
    node_index_1[:,1]+=1
    node_index_1[:,2]+=2

    node_index_0=torch.zeros(coo.shape[0],27,3,device=device)
    for t in [0,1,2]:
        node_index_0[:,:,t]=index[coo[:,0],coo[:,1],coo[:,2]].unsqueeze(1).repeat(1, 27)*3+t

    node_index_1=node_index_1.reshape(-1,1).repeat(1, 3).reshape(-1,3,3).transpose(-1,-2).reshape(1,-1)
    node_index_0=node_index_0.reshape(-1,1).repeat(1, 3).reshape(1,-1)

    K_value=K_value.reshape(-1)
    # F_value=F_value.reshape(-1)
    K = torch.sparse_coo_tensor(torch.cat([node_index_0,node_index_1],dim=0), K_value, (3 * coo.shape[0], 3 * coo.shape[0]), device=device).coalesce()
    # F = F_value.reshape(-1,3,6).view(-1,6)
    return K