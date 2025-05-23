import os
import torch
import random
import argparse
from omegaconf import OmegaConf
from train.dataset import DataLoader
from train.multigrid import *
from train.mink_lightning import *
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning as pl

# torch.autograd.set_detect_anomaly(True)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('medium')
torch.set_printoptions(
    precision=6,    # ç²¾åº¦ï¼ä¿çå°æ°ç¹åå ä½ï¼é»è®¤4
    threshold=200,
    edgeitems=100,
    linewidth=150,  # æ¯è¡æå¤æ¾ç¤ºçå­ç¬¦æ°ï¼é»è®¤80ï¼è¶è¿åæ¢è¡æ¾ç¤º
    profile=None,
    sci_mode=True  # ç¨ç§å­¦ææ¯æ³æ¾ç¤ºæ°æ®ï¼é»è®¤True
)

def ph_net(cfg):

    setup_seed(0)
    
    # Logger
    tb_logger = pl.pytorch.loggers.TensorBoardLogger(
                                                    save_dir=os.path.join(cfg.train.output_path,cfg.train.logger.path),
                                                    version=cfg.train.logger.version
                                                    )
    
    # Check point
    ckpt_callback= ModelCheckpoint(
        monitor="error", 
        dirpath=os.path.join(cfg.train.output_path,cfg.train.checkpoint.path),
        filename='epoch{epoch:02d}-error{error:.6f}',
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
        mode='min'
        )

    pl_module = MinkowskiLightningModule(cfg)
    # checkpoint = torch.load(cfg.pre_train, map_location="cpu")

    # pl_module.load_state_dict(checkpoint["state_dict"])
    # .load_from_checkpoint(cfg.pre_train)
    trainer = pl.Trainer(
                        max_epochs=cfg.train.max_epoch, 
                        check_val_every_n_epoch=5,
                        accelerator=cfg.device, 
                        strategy="ddp",
                        devices="auto",
                        logger=tb_logger,
                        log_every_n_steps=2,
                        # profiler="simple",
                        # enable_checkpointing=False,
                        callbacks=[ckpt_callback]
                        )
    
    if cfg.mode=='Train':
        train_dataloader=DataLoader(cfg.train.dataset_dir,cfg.train.num_works,cfg.train.batch_size)
        val_dataloaders=DataLoader(cfg.evaluation.dataset_dir,cfg.train.num_works,cfg.evaluation.batch_size)
        # val_dataloaders=DataLoader(cfg.evaluation.dataset_dir,cfg.train.num_works,cfg.evaluation.batch_size)
        if cfg.pre_train is not None:
            trainer.fit(pl_module,
                        train_dataloaders=train_dataloader,
                        val_dataloaders = val_dataloaders,
                        ckpt_path=cfg.pre_train
                    )
        else:
            trainer.fit(pl_module,
                        train_dataloaders=train_dataloader,
                        val_dataloaders = val_dataloaders,
                        # ckpt_path=cfg.pre_train
                    )
    if cfg.mode=='Test':
        test_dataloader=DataLoader(cfg.evaluation.dataset_dir,cfg.train.num_works,cfg.evaluation.batch_size)
        trainer.test(model=pl_module,
                    dataloaders=test_dataloader,
                    ckpt_path=cfg.pre_train
                    )
    



if __name__=='__main__':
    
    parser=argparse.ArgumentParser(description="MH-Net dataset generator")

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--world_size', type=int, default=4,help='Number of visible GPU')
    args=parser.parse_args()
    
    cfg=OmegaConf.load(args.config)
    ph_net(cfg)