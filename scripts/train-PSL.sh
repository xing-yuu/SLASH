# export OMP_NUM_THREADS=32
# CUDA_DEVICE_ORDER=PCI_BUS_ID
# CUDA_VISIBLE_DEVICES=0 python main.py configs/train.yaml --world_size=1
CUDA_VISIBLE_DEVICES=0 python main.py configs/train-PSL.yaml --world_size=8
# CUDA_VISIBLE_DEVICES=0 python main.py configs/psl-five.yaml --world_size=8
 