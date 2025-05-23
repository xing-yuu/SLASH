import torch
import MinkowskiEngine as ME
import time
from train.multigrid import calculate_legal_coordinates,node_KF_assembly,calculate_level_K_list,calculate_residual_2D
from train.numerical_homogenization import *
from train.loss_vis import *
from train.minimal_potential_energy_loss import MPE_full_all_batch


@torch.no_grad()   
def validate(net,K_list,f_batch_list,level_coo,batch,batch_size,level,smooth,r):
    """computer batch error between predicated homogenized tensor and the groundtruth

    Args:
        output (torch.cuda.FloatTensor): predicated displacement,it should be batch_size*18*N*N*N
        
        Ke (torch.cuda.FloatTensor): element stiffness matrics, it should be batch_size*24*24
        Fe (torch.cuda.FloatTensor): element force matrics, it should be batch_size*24*6
        X0 (torch.cuda.FloatTensor): element force matrics, it should be batch_size*24*6

    Returns:
        batch_error_mean,
        batch_error,
        batch_loss_actual,
        batch_predict
    """
    device=level_coo[0].device

    net.train()
    U_=net.infer(batch,level_coo,K_list,f_batch_list,smooth,batch_size)
    
    output_mink=ME.SparseTensor(
                    coordinates=level_coo[0].float(),
                    features=U_
                )
    
    output,_,_=output_mink.dense(min_coordinate=torch.IntTensor([0,0,0]))
    output=output[:,:,0:r,0:r,0:r]

    homo_num=numerical_homogenization(r,r,r,1,1,1,device)
    batch_predict=torch.zeros(batch_size,6,6,dtype=torch.float32,device=device)
    batch_actual=torch.zeros(batch_size,6,6,dtype=torch.float32,device=device)
    batch_error=torch.zeros(batch_size,dtype=torch.float32,device=device)
    U=torch.zeros(batch_size,18,r,r,r,dtype=torch.float32,device=device)
    for i in range(batch_size):
        # U[i]=homo_num.solve_by_torch(batch["voxel"][i], batch["Ke"][i],batch["Fe"][i])
        batch_predict[i]=homo_num.homogenized(batch["voxel"][i],output[i],batch["Ke"][i],batch["X0"][i])
        batch_actual[i]=batch["CH_GT"][i]
        batch_error[i]=torch.dist(batch_actual[i],batch_predict[i])/torch.linalg.norm(batch_actual[i])
    # U_sp=U[level_coo[0][:,0],:,level_coo[0][:,1],level_coo[0][:,2],level_coo[0][:,3]]
    # batch_loss_predict,_=MPE_full_all_batch(U_sp,K_list[0],f_batch_list[0],level_coo[0],batch_size)
    # residual_predict=torch.linalg.norm(calculate_residual_2D(node_stiffness,output_mink,node_load,r,batch_size))
    residual_predict=torch.linalg.norm(calculate_residual_2D(level_coo[0].float(),K_list[0],output_mink.F,f_batch_list[0],batch_size))

    # residual_gt=torch.linalg.norm(calculate_residual_2D(level_coo[0].float(),K_list[0],U_sp,f_batch_list[0],batch_size))

    
    validate_dict={}
    # validate_dict["t1"]=t1/batch_size
    # validate_dict["t2"]=t2/batch_size
    # validate_dict["t3"]=t3/batch_size
    # validate_dict["t4"]=t4/batch_size
    # validate_dict["t5"]=t5/batch_size
    # validate_dict["ta"]=ea/batch_size
    # validate_dict["tb"]=eb/batch_size
    # validate_dict["tss"]=tss/batch_size
    # validate_dict["tmm"]=tmm/batch_size
    # validate_dict["smooth_time"]=smooth_time/batch_size
    # validate_dict["ASS_TIME"]=ASS_TIME
    # validate_dict["INFER_TIME"]=INFER_TIME
    # validate_dict["encoder"]=encoder
    # validate_dict["unet"]=unet
    # validate_dict["decoder"]=decoder
    # validate_dict["correct"]=correct
    # print('batch_loss_predict',batch_loss_predict)
    # print('batch_resd_predict',residual_gt)
    validate_dict["error"]=batch_error.mean()
    validate_dict["error_max"]=torch.max(batch_error)
    validate_dict["error_min"]=torch.min(batch_error)
    validate_dict["residual_predict"]=residual_predict
    # validate_dict["time_init"]=t1
    # validate_dict["time_pre"]=t2

    return validate_dict
