# CUDA_VISIBLE_DEVICES=0 python datagen/generate.py --config=configs/dataset-train.yaml
# CUDA_VISIBLE_DEVICES=3 python DOF.py --config=configs/dataset-train.yaml
# CUDA_VISIBLE_DEVICES=4 python main.py configs/train.yaml --world_size=1
CUDA_VISIBLE_DEVICES=0 python datagen/generate-ISINS.py --config=configs/dataset-512.yaml