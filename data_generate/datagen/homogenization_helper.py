import numpy as np
import torch

class homogenization_helper:

    def __init__(self, cache_file, device) -> None:

        cache = np.load(cache_file)
        self.partial_N = torch.from_numpy(cache['partial_N']).to(device)
        self.weights = torch.from_numpy(cache['weight']).to(device)
        self.device = device

 

    def jacobian_matrix(self, hex, volume):
        # coumpute jacobian 27 * 3 * 3
        J = torch.matmul(self.partial_N, hex)
        # compute weight 27*1
        w = (J.det().unsqueeze(1)*self.weights).unsqueeze(2)
        return (J*w).sum(0)*2/volume

    def stiffness_force(self, hex, elastic_tensor):
        """compute the stiffness and force of a hex element 

        Args:
            hex ([type]): 8 * 3
            elastic_tensor ([type]): 6 * 6

        Returns:
            stiffness [type]: 24*24
            force [type] 24*6
        """

        # coumpute jacobian 27 * 3 * 3
        J = torch.matmul(self.partial_N.type_as(elastic_tensor), hex).to(self.device)

        # compute weight 27*1
        w = (J.det().unsqueeze(1)*self.weights).unsqueeze(2).type_as(elastic_tensor)

        # compute the inverse of partial N 27 * 3 * 8
        inv_J = J.inverse()
        inv_N = (inv_J@self.partial_N.type_as(elastic_tensor))

        # assigned local geometry matrix B 27*6*24
        B = inv_N.unsqueeze(1).repeat(1, 6, 1, 1)*0
        idx_B = torch.as_tensor([[0, 1, 2, 3, 3, 4, 4, 5, 5], [
                                0, 1, 2, 0, 1, 1, 2, 0, 2]], dtype=torch.int64, device=hex.device)
        idx_inv_N = torch.as_tensor(
            [0, 1, 2, 1, 0, 2, 1, 2, 0], dtype=torch.int64, device=hex.device)
        B[:, idx_B[0], idx_B[1], :] = inv_N[:, idx_inv_N, :]
        
        B = B.transpose(2, 3).contiguous().reshape(27, 6, 24)#27,6,3,8->27,6,8,3->27,6,24
        # print(B.transpose(1, 2).shape,elastic_tensor.shape,(B.transpose(1, 2)@elastic_tensor).shape,(elastic_tensor@B).shape)

        # Gaussian integration
        # k=B^T @ C @ B *w

        stiffness = (B.transpose(1, 2)@elastic_tensor@B*w).sum(0)
        force = (B.transpose(1, 2)@elastic_tensor*w).sum(0)
        # print((B.transpose(1, 2)@elastic_tensor*w).shape)
        # exit()
        return stiffness, force

    def macro_deformation(self, lx, ly, lz, elastic_tensor):
        hex = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [
                           1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=elastic_tensor.dtype, device=self.device)
        scale = torch.tensor(
            [lx, ly, lz], dtype=elastic_tensor.dtype, device=self.device)
        hex = torch.div(hex, scale)

        idx = torch.ones(24, dtype=torch.bool, device=self.device)
        idx[[0, 1, 2, 4, 5, 11]] = False

        ke, fe = self.stiffness_force(hex, elastic_tensor)

        X0 = torch.zeros(24, 6, dtype=elastic_tensor.dtype, device=self.device)
        X0[idx, :] = torch.inverse(ke[idx, :][:, idx])@fe[idx, :]

        return ke, X0
    def voxel_XKF(self, lx, ly, lz, elastic_tensor):
        hex = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [
                           1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=elastic_tensor.dtype, device=self.device)
        scale = torch.tensor(
            [lx, ly, lz], dtype=elastic_tensor.dtype, device=self.device)
        hex = torch.div(hex, scale)

        idx = torch.ones(24, dtype=torch.bool, device=self.device)
        idx[[0, 1, 2, 4, 5, 11]] = False

        ke, fe = self.stiffness_force(hex, elastic_tensor)

        X0 = torch.zeros(24, 6, dtype=elastic_tensor.dtype, device=self.device)
        X0[idx, :] = torch.inverse(ke[idx, :][:, idx])@fe[idx, :]

        return  X0,ke,fe


def isotropic_elastic_tensor(E, v):
    Lambda = v / (1. + v) / (1 - 2. * v)*E
    Mu = 1. / (2.*(1. + v))*E
    # print(Lambda,Mu)
    # exit()
    return torch.as_tensor([
        [Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
        [Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
        [Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
        [0, 0, 0, Mu, 0, 0],
        [0, 0, 0, 0, Mu, 0],
        [0, 0, 0, 0, 0, Mu]])


def shape_material_transform(material, J):
    if J.dim() == 2:
        MA = torch.tensor([[torch.square(J[0, 0]), torch.square(J[0, 1]), torch.square(J[0, 2]),
                            J[0, 0] * J[0, 1], J[0, 1] * J[0, 2], J[0, 0] * J[0, 2]],
                           [torch.square(J[1, 0]), torch.square(J[1, 1]), torch.square(J[1, 2]),
                            J[1, 0] * J[1, 1], J[1, 1] * J[1, 2], J[1, 0] * J[1, 2]],
                           [torch.square(J[2, 0]), torch.square(J[2, 1]), torch.square(J[2, 2]),
                            J[2, 0] * J[2, 1], J[2, 1] * J[2, 2], J[2, 0] * J[2, 2]],
                           [2 * J[0, 0] * J[1, 0], 2 * J[0, 1] * J[1, 1],
                            2 * J[0, 2] * J[1, 2], J[0, 0] *
                            J[1, 1] + J[0, 1] * J[1, 0],
                            J[0, 1] * J[1, 2] + J[0, 2] * J[1, 1],
                            J[0, 0] * J[1, 2] + J[0, 2] * J[1, 0]],
                           [2 * J[1, 0] * J[2, 0], 2 * J[1, 1] * J[2, 1],
                            2 * J[1, 2] * J[2, 2], J[1, 0] *
                            J[2, 1] + J[1, 1] * J[2, 0],
                            J[1, 1] * J[2, 2] + J[1, 2] * J[2, 1],
                            J[1, 0] * J[2, 2] + J[1, 2] * J[2, 0]],
                           [2 * J[2, 0] * J[0, 0], 2 * J[2, 1] * J[0, 1],
                            2 * J[2, 2] * J[0, 2], J[2, 0] *
                            J[0, 1] + J[2, 1] * J[0, 0],
                            J[2, 1] * J[0, 2] + J[2, 2] * J[0, 1],
                            J[2, 0] * J[0, 2] + J[2, 2] * J[0, 0]],
                           ], dtype=material.dtype, device=material.device)
        return MA.t() @ material @ MA

    elif J.dim() == 3:
        MA = torch.empty(J.shape[0], 6, 6,
                         dtype=material.dtype, device=material.device)
        MA[:, 0, :] = torch.vstack([torch.square(J[:, 0, 0]), torch.square(J[:, 0, 1]), torch.square(
            J[:, 0, 2]), J[:, 0, 0] * J[:, 0, 1], J[:, 0, 1] * J[:, 0, 2], J[:, 0, 0] * J[:, 0, 2]]).t()
        MA[:, 1, :] = torch.vstack([torch.square(J[:, 1, 0]), torch.square(J[:, 1, 1]), torch.square(
            J[:, 1, 2]), J[:, 1, 0] * J[:, 1, 1], J[:, 1, 1] * J[:, 1, 2], J[:, 1, 0] * J[:, 1, 2]]).t()
        MA[:, 2, :] = torch.vstack([torch.square(J[:, 2, 0]), torch.square(J[:, 2, 1]), torch.square(
            J[:, 2, 2]), J[:, 2, 0] * J[:, 2, 1], J[:, 2, 1] * J[:, 2, 2], J[:, 2, 0] * J[:, 2, 2]]).t()
        MA[:, 3, :] = torch.vstack([2 * J[:, 0, 0] * J[:, 1, 0], 2 * J[:, 0, 1] * J[:, 1, 1], 2 * J[:, 0, 2] * J[:, 1, 2], J[:, 0, 0] * J[:, 1, 1] +
                                   J[:, 0, 1] * J[:, 1, 0], J[:, 0, 1] * J[:, 1, 2] + J[:, 0, 2] * J[:, 1, 1], J[:, 0, 0] * J[:, 1, 2] + J[:, 0, 2] * J[:, 1, 0]]).t()
        MA[:, 4, :] = torch.vstack([2 * J[:, 1, 0] * J[:, 2, 0], 2 * J[:, 1, 1] * J[:, 2, 1], 2 * J[:, 1, 2] * J[:, 2, 2], J[:, 1, 0] * J[:, 2, 1] +
                                   J[:, 1, 1] * J[:, 2, 0], J[:, 1, 1] * J[:, 2, 2] + J[:, 1, 2] * J[:, 2, 1], J[:, 1, 0] * J[:, 2, 2] + J[:, 1, 2] * J[:, 2, 0]]).t()
        MA[:, 5, :] = torch.vstack([2 * J[:, 2, 0] * J[:, 0, 0], 2 * J[:, 2, 1] * J[:, 0, 1], 2 * J[:, 2, 2] * J[:, 0, 2], J[:, 2, 0] * J[:, 0, 1] +
                                   J[:, 2, 1] * J[:, 0, 0], J[:, 2, 1] * J[:, 0, 2] + J[:, 2, 2] * J[:, 0, 1], J[:, 2, 0] * J[:, 0, 2] + J[:, 2, 2] * J[:, 0, 0]]).t()

        return MA.transpose(1, 2) @ material @ MA

def cell_dof_indices(nelx, nely, nelz):
    nodeidx = torch.zeros(1+nelx, 1+nely, 1+nelz,
                            dtype=torch.long)
    nodeidx[:-1, :-1, :-1] = torch.arange(
        nelx*nely*nelz).reshape(nelx, nely, nelz)
    nodeidx[-1, :, :] = nodeidx[0, :, :]
    nodeidx[:, -1, :] = nodeidx[:, 0, :]
    nodeidx[:, :, -1] = nodeidx[:, :, 0]

    cellidx = torch.empty(
        nelx*nely*nelz, 8, dtype=torch.long)
    cellidx[:, 0] = nodeidx[:-1, :-1, :-1].reshape(-1)
    cellidx[:, 1] = nodeidx[1:, :-1, :-1].reshape(-1)
    cellidx[:, 2] = nodeidx[1:, 1:, :-1].reshape(-1)
    cellidx[:, 3] = nodeidx[:-1, 1:, :-1].reshape(-1)
    cellidx[:, 4] = nodeidx[:-1, :-1, 1:].reshape(-1)
    cellidx[:, 5] = nodeidx[1:, :-1, 1:].reshape(-1)
    cellidx[:, 6] = nodeidx[1:, 1:, 1:].reshape(-1)
    cellidx[:, 7] = nodeidx[:-1, 1:, 1:].reshape(-1)

    offset = torch.tensor([0, 1, 2], dtype=torch.long).repeat(8)
    return torch.repeat_interleave(cellidx, 3, dim=1)*3+offset

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
            K_node: [shape: 256*3*81], Node stiffness matrix 
            F_node: [shape: 256*3*6], Node load
    """

    K_node = torch.zeros(256,81,3,dtype=torch.float,device=Ke.device)
    F_node = torch.zeros(256,3,6,dtype=torch.float,device=Fe.device)
    
    # Assembly of 8 single voxels
    id=1
    # all_nodelist: Which eight of the 27 nodes are affected by each voxel
    all_nodelist=[
                [0,9,12,3,1,10,13,4],
                [1,10,13,4,2,11,14,15],
                [3,12,15,6,4,13,16,7],
                [4,13,16,7,5,14,17,8],
                [9,18,21,12,10,19,22,13],
                [10,19,22,13,11,20,23,14],
                [12,21,24,15,13,22,25,16],
                [13,22,25,16,14,23,26,17]
                ]
    
    # target: Which direction is the target node in 8 voxels
    target=[6,2,5,1,7,3,4,0]
    for nodeid in range(0,8):
        nodelist=all_nodelist[nodeid]
        for i in range(0,8):
            K_node[id][nodelist[i]*3:nodelist[i]*3+3]=Ke[i*3:i*3+3,target[nodeid]*3:target[nodeid]*3+3]
        id=id<<1
    id=1
    for i in range(0,8):     
        F_node[id]=Fe[target[i]*3:target[i]*3+3]
        id=id<<1
        
    # F_node=F_node.reshape(-1,6).cpu()
    # data = pd.DataFrame(F_node)
    # writer = pd.ExcelWriter('F_node_init.xlsx')		# 写入Excel文件
    # data.to_excel(writer, 'page_1')
    # # , float_format='%.6f')		# ‘page_1’是写入excel的sheet名
    # writer._save()
    # exit()
    # multi
    std=[128,64,32,16,8,4,2,1]
    for i in range (1,256):
        tips=i
        if not torch.all(K_node[i] == 0).item():
            continue
        for bin in std:
            if tips>=bin:
                K_node[i]+=K_node[bin]
                F_node[i]+=F_node[bin]
                tips-=bin
    K_node=K_node.transpose(1,2).contiguous()    
    return K_node,F_node