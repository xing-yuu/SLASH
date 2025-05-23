CUDA_VISIBLE_DEVICES=6 python datagen/generate.py --config=configs/dataset-test.yaml  
export OMP_NUM_THREADS=32
# CUDA_VISIBLE_DEVICES=4 python main.py configs/train.yaml --world_size=1
# CUDA_VISIBLE_DEVICES=0 python datagen/generate.py --config=configs/dataset-train.yaml