# SLASH: A Spectral-Aligned Numerical Neural Hybrid Solver for Microstructure Homogenization

This repository contains the official implementation of the paper:
**"SLASH: A Spectral-Aligned Numerical Neural Hybrid Solver for Microstructure Homogenization"**
We provide the training code, configuration files, and scripts to reproduce our results on periodic microstructure datasets.
The dataset is not publicly released; please contact us if you have a legitimate need for access.

## 1. Environment Setup  
To set up the Python environment, please install the required packages:  
```bash  
pip install -r requirements.txt  
```  

### Hardware Requirements  
- Training was conducted on **8 NVIDIA A100 GPUs**.  
- Multi-GPU training with PyTorch Lightning is supported.  

## 2. Configuration File Overview  
We use `.yaml` configuration files to define training parameters. Below is a breakdown of the important variables in `scripts/train-TPMS.yaml`:  

| Key | Description |  
|-----|-------------|  
| `mode` | code execution mode. `train` indicates training mode, `test` indicates testing mode (requires a pre-trained model). |  
| `pre_train` | Path to the pre-trained model. |  
| `model.resolution` | Resolution of the microstructure. |  
| `model.level` | Number of multi-level stages in the SLASH. |  
| `model.smooth` | number of iterations for the PCG numerical solver. |  


## 3. Quick Start  
To start training on the TPMS dataset, run the following shell script:  

```bash  
sh scripts/train-TPMS.sh  
```  

Please cite our paper if you use this code in your research.  
For any issues or questions, feel free to open an issue or contact the authors.  
