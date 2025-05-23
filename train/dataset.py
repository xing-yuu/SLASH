import torch
import os
import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME

def minkowski_collate_fn(list_data):
    """
    Collation function for MinkowskiEngine.SparseTensor that creates batched
    cooordinates given a list of dictionaries.
    input,voxel,node,node_stiffness,node_load,Ke,Fe,X0
    """

    coordinates_batch,node_type_batch= ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["node_type"] for d in list_data],
        dtype=torch.float32,
    )
    voxel_batch=torch.stack([d["voxel"] for d in list_data])
    CH_GT_batch=torch.stack([d["CH_GT"] for d in list_data])
    Ke_batch=torch.stack([d["Ke"] for d in list_data])
    Fe_batch=torch.stack([d["Fe"] for d in list_data])
    X0_batch=torch.stack([d["X0"] for d in list_data])
    K_node_batch=torch.stack([d["K_node"] for d in list_data])
    F_node_batch=torch.stack([d["F_node"] for d in list_data])
    K_node_soft_batch=torch.stack([d["K_node_soft"] for d in list_data]) 
    F_node_soft_batch=torch.stack([d["F_node_soft"] for d in list_data])
    C_H_batch=torch.stack([d["C_H"] for d in list_data])
    C_H_soft_batch=torch.stack([d["C_H_soft"] for d in list_data])
    return {
        "coordinates": coordinates_batch,
        "node_type": node_type_batch.long(),
        "voxel": voxel_batch.long(),
        "CH_GT": CH_GT_batch,
        "Ke": Ke_batch,
        "Fe": Fe_batch,
        "X0": X0_batch,
        "K_node": K_node_batch,
        "F_node": F_node_batch,
        "K_node_soft": K_node_soft_batch,
        "F_node_soft": F_node_soft_batch,
        "C_H": C_H_batch,
        "C_H_soft": C_H_soft_batch,
    }


class DataSet(torch.utils.data.Dataset):
    def __init__(self,base_path_list):
        self.data_name_set=[]

        for base_path in base_path_list:
            datanames = os.listdir(base_path)
            pbar = tqdm(datanames)
            for dataname in pbar:
                pbar.set_description('Data Loading')
                if os.path.splitext(dataname)[1] == '.npz' and dataname!='material.npz':
                    self.data_name_set.append(os.path.join(base_path,dataname))
                    
                    

    def __len__(self):
        # return 2
        return len(self.data_name_set)
    
    def __getitem__(self, index):
        data=np.load(self.data_name_set[index],allow_pickle=True)
        return {
            "coordinates":torch.from_numpy(data["coordinates"]),
            "node_type":torch.from_numpy(data["node_type"]),
            "voxel":torch.from_numpy(data["voxel"]),
            "CH_GT":torch.from_numpy(data["CH_GT"]),
            "Ke":torch.from_numpy(data["Ke"]),
            "Fe":torch.from_numpy(data["Fe"]),
            "X0":torch.from_numpy(data["X0"]),
            "K_node":torch.from_numpy(data["K_node"]),
            "F_node":torch.from_numpy(data["F_node"]),
            "K_node_soft":torch.from_numpy(data["K_node_soft"]),
            "F_node_soft":torch.from_numpy(data["F_node_soft"]),
            "C_H":torch.from_numpy(data["C_H"]),
            "C_H_soft":torch.from_numpy(data["C_H_soft"]),
        }

def DataLoader(path,num_workers,batch_size):
    dataset=DataSet(path)
    return torch.utils.data.DataLoader(
            dataset,
            num_workers = num_workers,
            batch_size=batch_size,
            collate_fn=minkowski_collate_fn,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )