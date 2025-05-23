eval "$(conda shell.bash hook)"
conda activate A100

cd /MinkowskiEngine
export CUDA_HOME=/usr/local/cuda-11.7
export CXX=g++-7

python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

pip install omegaconf
pip install einops

cd $ISINS_DIR

# export NCCL_P2P_DISABLE=1

sh scripts/train-Truss.sh