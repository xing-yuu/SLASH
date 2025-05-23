import torch

def assembly_node_matrix(Ke,Fe):
    """assemble the node stiffness matrix by element stiffness matrix

        The assembly of the node stiffness matrix is only related to the presence of voxels of the 
        target node and the surrounding 8 ocuraries, so there are 255 cases. 
        Each node has three degrees of freedom, and there are 27 nodes around it, 
        for a total of 81 degrees of freedom.
        
        Args:
            Ke: [shape: 24*24], Element stiffness matrix 
            Fe: [shape: 24*6]: Element load

        Returns:
            K_node: [shape: 256,27,3,3], Node stiffness matrix 
            F_node: [shape: 256,3,6], Node load
    """
    # print(Fe.shape)
    Ke_2=Ke.reshape(8,3,8,3).contiguous()

    Fe_2=Fe.reshape(8,3,6)
    # Fe_2=Fe.reshape(3,8,6).transpose(0,1)
    # print(torch.sum(Fe_2,dim=0))
    # print(Fe_2[0]+Fe_2[1]+Fe_2[2]+Fe_2[3])
    # print(Fe_2[0]+Fe_2[1]+Fe_2[5]+Fe_2[4])
    # print(Fe_2[0]+Fe_2[3]+Fe_2[4]+Fe_2[7])
    # exit()
    K_node = torch.zeros(256,27,3,3,dtype=torch.float,device=Ke.device)
    F_node = torch.zeros(256,3,6,dtype=torch.float,device=Ke.device)
    
    id=1
    all_nodelist=[
                [0,9,12,3,1,10,13,4],
                [1,10,13,4,2,11,14,5],
                [3,12,15,6,4,13,16,7],
                [4,13,16,7,5,14,17,8],
                [9,18,21,12,10,19,22,13],
                [10,19,22,13,11,20,23,14],
                [12,21,24,15,13,22,25,16],
                [13,22,25,16,14,23,26,17]]
    
    # target: Which direction is the target node in 8 voxels
    target=[6,2,5,1,7,3,4,0]
    for eid in range(0,8):
        nodelist=all_nodelist[eid]
        for i in range(0,8):
            K_node[id,nodelist[i],:,:]=Ke_2[target[eid],:,i,:]
        id=id*2
        
    id=1
    for eid in range(0,8):
        F_node[id,:,:]+=Fe_2[target[eid],:,:]
        id=id*2

    std=[128,64,32,16,8,4,2,1]
    for i in range (1,256):
        tips=i
        if i in std:
            continue
        for bin in std:
            if tips>=bin:
                K_node[i,:,:,:]=K_node[i,:,:,:]+K_node[bin,:,:,:]
                F_node[i,:,:]=F_node[i,:,:]+F_node[bin,:,:]
                tips-=bin
    # print('255',F_node[255])
    # exit()
    return K_node,F_node

def get_node_type(voxel):
    """
    According to the 8 ocuraries around the node, the node type is judged
        Args:
            voxel shape(n , n , n)
            0,1 indicates whether it is a solid voxel
            
        Returns:
            voxel (long)
            shape(n , n , n)
                [0,255]indicates node type
    """
    node_type=torch.zeros(voxel.shape,dtype=torch.int,device=voxel.device)
    hex=torch.tensor([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]],dtype=torch.int,device=voxel.device)
    r=voxel.shape[1]

    soile_voxel_coo=torch.nonzero(voxel,as_tuple=False).to(voxel.device)
    tips=1
    for i in range (0,8):
        coo=((soile_voxel_coo+hex[i]+r)%r).to(voxel.device)
        node_type[coo[:,0],coo[:,1],coo[:,2]]+=tips
        tips=tips*2

    return node_type.long()


def assemble_stiffness(voxel,K_node,F_node):
    """compute the stiffness and force of a hex element 

        Args:
            voxel
                hex ([type]): shape(n , n , n)
            Ke   
                element stiffness ([type]): shape(24 , 24)
            Fe   
                element load ([type]): shape(24 , 6)

        Returns:
            node_stiffness 
                shape(r,r,r,27*3*3)
            node_load
                shape(r,r,r,27*3*6)
    """

    r=voxel.shape[0]
    node_type=get_node_type(voxel).reshape(-1)
    node_stiffness=K_node[node_type,:,:,:]
    node_load=F_node[node_type,:,:]
    return node_stiffness.reshape(r,r,r,-1),node_load.reshape(r,r,r,-1)

def delete_anchor(node_type,Anchor):
    '''
        Args:
            node_type_hard
            Anchor
        Returns:
            node_type
    '''

    hex=torch.tensor([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]],dtype=torch.int,device=Anchor.device)
    r=node_type.shape[1]

    tips=1
    for i in range (0,8):
        coo=((Anchor+hex[i]+r)%r)
        node_type[coo[:,0],coo[:,1],coo[:,2]]-=tips
        tips=tips*2
    return node_type.long()
