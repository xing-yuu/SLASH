import torch
import torch.sparse as ts
from torch_sparse import coalesce
from linear_operator.utils.linear_cg import linear_cg
from homogenization_helper import *
import numpy as np
import pandas as pd

class numerical_homogenization:

    # rewrited by penghao

    def __init__(self, nelx, nely, nelz, lx, ly, lz, device) -> None:

        self.device=device
        self.__lx = lx
        self.__ly = ly
        self.__lz = lz
        self.__nelx = nelx
        self.__nely = nely
        self.__nelz = nelz
        nel = nelx * nely * nelz
        nodeidx = torch.arange(nel, device=self.device).view(nelx, nely, nelz)

        index = torch.as_tensor([0], device=self.device)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 0, index)), 0)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 1, index)), 1)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 2, index)), 2)

        node_list = [nodeidx[0:nelx, 0:nely, 0:nelz].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 0:nely, 0:nelz].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 1:nely + 1, 0:nelz].reshape((nel, 1)),
                     nodeidx[0:nelx, 1:nely + 1, 0:nelz].reshape((nel, 1)),
                     nodeidx[0:nelx, 0:nely, 1:nelz + 1].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 0:nely, 1:nelz + 1].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 1:nely + 1,
                     1:nelz + 1].reshape((nel, 1)),
                     nodeidx[0:nelx, 1:nely + 1, 1:nelz + 1].reshape((nel, 1))]

        self.__cellidx = torch.zeros(
            8, 3, nel, device=self.device, dtype=torch.int64)
        self.__cellseq = torch.zeros(nel, 8, device=self.device, dtype=torch.int64)

        for i in range(8):
            self.__cellidx[i] = self.index2xyz(node_list[i])
            self.__cellseq[:, i] = node_list[i].view(-1)
            
        self.__nodeidx = nodeidx

    def printindex(self):
        print(self.__cellidx.size())
        return self.__cellidx

    def index2xyz(self, index):
        x = index.div(self.__nely * self.__nelz, rounding_mode='floor')
        temp = index .remainder(self.__nely * self.__nelz)
        y = temp.div(self.__nely,rounding_mode='floor')
        z = temp.remainder( self.__nely)
        xyz = torch.cat((x, y, z), 1)
        return xyz.t()

    # rewrited by penghao
    def solid_cell(self, voxel):
        """return solid cell index from voxel

        Args:
            voxel (torch.cuda.FloatTensor): it should be N*N*N 

        Returns:
            solid_cell (torch.cuda.IntTensor): it should be 8*3*nel 
            ancher_cell(torch.cuda.FloatTensor): it should be a 
            K_mask(torch.cuda.FloatTensor): it should be a*24*24 
            F_mask(torch.cuda.FloatTensor): it should be a*24*6 
        """
        # print(voxel.device)
        voxelidx = torch.arange(0, voxel.numel(), device=self.device)
        voxelidx = torch.masked_select(
            voxelidx, voxel.type(torch.bool).contiguous().view(-1))
        solid_cell = self.__cellidx[:, :, voxelidx]
        solid_seq = self.__cellseq[voxelidx, :]
        anchor = self.__nodeidx[solid_cell[0, 0, 0],
                                solid_cell[0, 1, 0], solid_cell[0, 2, 0]]
        # print(solid_cell[0, 0, 0], solid_cell[0, 1, 0], solid_cell[0, 2, 0])
        # print(anchor)
        mask = torch.eq(solid_seq, anchor)

        anchor_index = torch.arange(
            0, voxelidx.numel(), device=self.device)
        anchor_cell = anchor_index.masked_select(mask.sum(1).type(torch.bool))
        anchor_mask = mask[anchor_cell, :]

        if anchor_mask.dim() == 1:
            anchor_mask.unsqueeze_(0)

        anchor_mask = ~anchor_mask.unsqueeze(2).repeat(
            1, 1, 3).reshape(-1, 24).unsqueeze(2)
        anchor_mask = torch.as_tensor(
            anchor_mask, dtype=torch.float64, device=self.device)
        K_mask = anchor_mask.bmm(anchor_mask.transpose(1, 2))
        K_diag = torch.eye(24, 24, dtype=K_mask.dtype, device=self.device).unsqueeze(
            0).repeat(K_mask.shape[0], 1, 1)
        K_mask = torch.logical_or(K_mask, K_diag)
        F_mask = anchor_mask.repeat(1, 1, 6)
        return solid_cell, anchor_cell, K_mask, F_mask

   
    def hexahedron(self, C):
        dx = self.__lx / self.__nelx / 2
        dy = self.__ly / self.__nely / 2
        dz = self.__lz / self.__nelz / 2

        pp = torch.as_tensor(
            [-pow(3 / 5, 0.5), 0, pow(3 / 5, 0.5)], dtype=C.dtype, device=self.device)
        ww = torch.as_tensor([5 / 9, 8 / 9, 5 / 9],
                             dtype=C.dtype, device=self.device)
        Ke = torch.zeros(24, 24, dtype=C.dtype, device=self.device)
        Fe = torch.zeros(24, 6, dtype=C.dtype, device=self.device)

        dxdydz = torch.as_tensor(
            [[-dx, dx, dx, -dx, -dx, dx, dx, -dx], [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
             [-dz, -dz, -dz, -dz, dz, dz, dz, dz]], dtype=C.dtype, device=self.device).t()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = pp[i]
                    y = pp[j]
                    z = pp[k]
                    qxqyqz = torch.as_tensor(
                        [[-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                          ((y + 1) * (z - 1)) / 8, ((y - 1) *
                                                    (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                          ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8],
                         [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                          ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) /
                          8, -((x + 1) * (z + 1)) / 8,
                          ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8],
                         [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                          ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) /
                          8, -((x + 1) * (y - 1)) / 8,
                          ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]], dtype=C.dtype, device=self.device)

                    J = qxqyqz @ dxdydz
                    invJ = torch.inverse(J)
                    qxyz = invJ @ qxqyqz
                    B = torch.zeros(6, 24, dtype=C.dtype, device=self.device)

                    for i_B in range(8):
                        B[:, i_B * 3:(i_B + 1) * 3] = torch.as_tensor(
                            [[qxyz[0, i_B], 0, 0],
                             [0, qxyz[1, i_B], 0],
                             [0, 0, qxyz[2, i_B]],
                             [qxyz[1, i_B],
                              qxyz[0, i_B], 0],
                             [0, qxyz[2, i_B],
                              qxyz[1, i_B]],
                             [qxyz[2, i_B], 0, qxyz[0, i_B]]], dtype=C.dtype,
                            device=self.device)

                    weight = J.det() * ww[i] * ww[j] * ww[k]

                    Ke = Ke + weight * B.transpose(0, 1) @ C @ B
                    Fe = Fe + weight * B.transpose(0, 1) @ C

        return Ke, Fe

    
    def homogenized(self, voxel, U, ke_hard, X0):
        solid_seq = self.__cellseq[voxel.type(
            torch.bool).contiguous().view(-1), :]
        n = solid_seq.shape[0]

        volume = self.__lx * self.__ly * self.__lz

        # reshape U(18,N,N,N)->u(6,3,N^3),then transpose u(6,3,N^3)->u(6,N^3,3) 
        u_ = U.contiguous().view(6, 3, -1).transpose(1, 2)
        # reshape u(6,N^3,3)->u(6,N^3*3),transpose u(6,N^3*3)->u(N^3*3,6)
        u_ = u_.contiguous().view(6, -1).t()

        # index_u=self.__cellidx.movedim(2,0).contiguous().view(-1).to(U.device)

        index_u = torch.empty(n, 8, 3, dtype=torch.int64, device=self.device)
        for i in range(3):
            index_u[:, :, i] = 3 * solid_seq + i
        index_u = index_u.contiguous().view(-1)

        u = u_[index_u, :].contiguous().view(n, 24, 6)
        del index_u

        CH = torch.zeros(6, 6, dtype=ke_hard.dtype, device=self.device)
        L = X0-u
        CH = torch.einsum('bij,ik,bkl-> jl',L,ke_hard,L)
        CH=1 / volume * CH
        # data = pd.DataFrame(CH)
        # writer = pd.ExcelWriter('CH.xlsx')		# 写入Excel文件
        # data.to_excel(writer, 'page_1', float_format='%.3f')		# ‘page_1’是写入excel的sheet名
        # writer._save()
        return CH


    def assembly(self, voxel, Ke, Fe):
        """assemble stiffness matrix K and force matrix for solid cell

        Args:
           voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
           Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
           Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 

        Returns:
            K(torch.cuda.Soarse_COO_Tensor)
            F(torch.cuda.FloatTensor)
        """
        voxelidx = torch.arange(0, voxel.numel(),device=self.device)
        voxelidx = torch.masked_select(
            voxelidx, voxel.type(torch.bool).contiguous().view(-1))

        solid_seq = self.__cellseq[voxelidx, :]

        _, anchor_cell, K_mask, F_mask = self.solid_cell(voxel)

        indices, inv_indices = torch.unique(
            solid_seq, sorted=True, return_inverse=True)
        # print(self.__cellseq)
        # print(voxelidx)
        # print(solid_seq)
        # print(inv_indices)
        # print(indices)
        # exit()
        nv = indices.shape[0]
        nc = inv_indices.shape[0]

        dof = torch.zeros(nv, 3, device=self.device)
        dof_indices = torch.zeros(
            nc, 24, device=self.device, dtype=torch.int64)

        for i in range(3):
            dof[:, i] = 3 * indices + i
            dof_indices[:, torch.arange(0, 8) * 3 + i] = inv_indices * 3 + i

        dof = torch.as_tensor(dof.contiguous().view(-1),
                              dtype=torch.int64, device=self.device)
        dof_indices = torch.as_tensor(
            dof_indices.unsqueeze(2), dtype=Ke.dtype, device=self.device)
        # print(dof_indices)
        vK = Ke.repeat(nc, 1, 1)
        vF = Fe.repeat(nc, 1, 1)
        # new anchor by set F(anchor)->0
        for idx in range(K_mask.shape[0]):
            vK[anchor_cell[idx], :, :] = vK[anchor_cell[idx], :, :] * \
                                         K_mask[idx, :, :]
            vF[anchor_cell[idx], :, :] = vF[anchor_cell[idx], :, :] * \
                                         F_mask[idx, :, :]

        # torch-sparse_solver version
        Kij = torch.zeros(2, 24 * 24 * inv_indices.shape[0], device=self.device)
        temp = torch.ones(
            inv_indices.shape[0], 24, dtype=Ke.dtype, device=self.device).unsqueeze(2)
        Kij[0, :] = dof_indices.bmm(temp.transpose(1, 2)).contiguous().view(-1)
        Kij[1, :] = temp.bmm(dof_indices.transpose(1, 2)).contiguous().view(-1)

        K = torch.sparse_coo_tensor(Kij, vK.contiguous(
        ).view(-1), (3 * nv, 3 * nv), device=self.device).coalesce()

        
        Kij,_=coalesce(torch.as_tensor(Kij,dtype=torch.int64),vK.contiguous(
        ).view(-1),3*nv,3*nv)

        # print(Kij.shape)
        # print(K.indices().shape)

        Fij = torch.zeros(2, 24 * 6 * inv_indices.shape[0], device=self.device)
        temp_F = torch.ones(
            inv_indices.shape[0], 6, dtype=Ke.dtype, device=self.device).unsqueeze(2)
        Fij[0, :] = dof_indices.bmm(
            temp_F.transpose(1, 2)).contiguous().view(-1)
        Fij[1, :] = torch.arange(0, 6, device=self.device).unsqueeze(0).unsqueeze(
            0).repeat(inv_indices.shape[0], 24, 1).contiguous().view(-1)
        # Fij=Fij.to(torch.int).T
        # data = pd.DataFrame(Fij)
        # writer = pd.ExcelWriter('Fij.xlsx')		# 写入Excel文件
        # data.to_excel(writer, 'page_1')
        # # , float_format='%.6f')		# ‘page_1’是写入excel的sheet名
        # writer._save()
        # exit()
        
        F = torch.sparse_coo_tensor(Fij, vF.contiguous(
        ).view(-1), (3 * nv, 6), device=self.device).coalesce().to_dense()

        return K, F, dof

    # def solve_by_amg(self, voxel, Ke, Fe, tol=1e-4, maxit=5000)
    def solve_by_torch(self, voxel, Ke, Fe, tol=1e-4, maxit=5000):
        """solve Ku=f by pytorch using linear conjugate gradient method

        Args:
            voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
            Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
            Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 
            tol (float, optional): Linear cg tolerance. Defaults to 1e-3.
            maxit (int, optional): Maximum iteration number. Defaults to 1000.

        Returns:
            U (torch.cuda.FloatTensor): it should be 18*N*N*N
        """

        K, F, dof = self.assembly(voxel, Ke, Fe)
        # np.savetxt('KK.csv',K.to_dense().cpu().numpy(), delimiter=",")
        # # np.savetxt('FF.csv',F.reshape(-1,3,6)[:,:,0].cpu().numpy(), delimiter=",")
        # exit()
        def Kmm(rhs): return ts.mm(K, rhs)

        X = linear_cg(Kmm, F, tolerance=tol, max_iter=maxit)
        X=X.reshape(-1,3,6).reshape(-1,18)
        X=X-torch.mean(X, dim=0).unsqueeze(0)
        X=X.reshape(-1,3,6).reshape(-1,6)
        # X = X-torch.mean(X, dim=0).unsqueeze(0)
        
        u = torch.zeros(3 * self.__cellseq.shape[0], 6,dtype=X.dtype,device=self.device)
        u[dof, :] = X

        # transpose u(6,N^3*3)<-u(N^3*3,6),reshape u(6,N^3,3)-<u(6,N^3*3)
        u = u.t().contiguous().view(6, -1, 3)
        
        # transpose u(6,3,N^3)<-u(6,N^3,3),reshape U(18,N,N,N)<-u(6,3,N^3),
        u = u.transpose(1, 2).contiguous().view(
            18, self.__nelx, self.__nely, self.__nelz)
        return u

    @torch.no_grad()
    def MPE_loss(self, voxel,Ke,Fe,input):
        """compute minimal potential energy according to 0.5*U^T@K@U-U^T@F
        Args:
           voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
           Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
           Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 
           U (torch.cuda.FloatTensor): it should be N*18 (to be test)

        Returns:
            energy(float) :  minimal potential energy
        """

        coo=torch.nonzero(voxel,as_tuple=False)
        elm_nbr=coo.shape[0]
        
        node_feature=torch.zeros(8,elm_nbr,18,dtype=torch.float,device=input.device)
        hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=input.device)
        
        for i in range (0,8):
            node_coo=(coo+hex[i])%self.__nelx
            node_feature[i,:,:]=input[:,node_coo[:,0],node_coo[:,1],node_coo[:,2]].t()


        node_feature=node_feature.transpose(0,1).reshape(-1,6,3).contiguous().transpose(1,2).contiguous()
        node_feature=node_feature.view(elm_nbr,8,3,6).contiguous().view(elm_nbr,24,6)
        
        Ku_F=(torch.einsum('ij,bjk->bik',Ke,node_feature)*0.5-Fe).unsqueeze(1)
        node_feature=node_feature.transpose(1,2).contiguous().unsqueeze(2)
        energy=torch.sum(torch.matmul(node_feature,Ku_F).view(elm_nbr,6,6)*torch.eye(6, 6, dtype=node_feature.dtype, device=self.device))

        return energy
def isotropic_elastic_tensor(E, v):
    Lambda = v / (1. + v) / (1 - 2. * v)*E
    Mu = 1. / (2.*(1. + v))*E

    return torch.as_tensor([
        [Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
        [Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
        [Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
        [0, 0, 0, Mu, 0, 0],
        [0, 0, 0, 0, Mu, 0],
        [0, 0, 0, 0, 0, Mu]])
if __name__=='__main__': 

    res=64
    sol=numerical_homogenization(res,res,res,1,1,1,'cuda:0')
    voxel=np.load('/mnt/local/HNET-ZJ-1009/datagen/top0.npy')
    voxel=torch.from_numpy(voxel).to('cuda:0')
    print(torch.sum(voxel)/(64**3))
    C_H=isotropic_elastic_tensor(1,0.35).to(torch.double).to('cuda:0')
    cache_file='cache/fem_cache.npz'
    homo_helper=homogenization_helper(cache_file,'cuda:0')
    X0,Ke,Fe=homo_helper.voxel_XKF(res,res,res,C_H)
    U=sol.solve_by_torch(voxel,Ke,Fe)
    ch=sol.homogenized(voxel, U, Ke, X0)
    
    print(ch)