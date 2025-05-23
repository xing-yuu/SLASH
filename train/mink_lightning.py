import os
import torch
import lightning as pl

from train.minimal_potential_energy_loss import *
from train.model import ISINS
from train.multigrid import calculate_legal_coordinates,calculate_level_K_list,fine2coarse_interpolation
# form train.multigrid import smooth_pcg
from train.eval import *
from train.loss_vis import *
import torch.optim as optim

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult    # cycle steps magnification
        self.base_max_lr = max_lr   # first max learning rate

        self.max_lr = max_lr    # max learning rate in the current cycle
        self.min_lr = min_lr    # min learning rate
        self.warmup_steps = warmup_steps    # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps    # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch     # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                                                    for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param=s_param.to(param.device)
                s_param.sub_(one_minus_decay * (s_param - param)) # only update the ema-params

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                                shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


# define the LightningModule
class MinkowskiLightningModule(pl.LightningModule):
    """
    Segmentation Module for MinkowskiEngine.
    """

    def __init__(
        self,
        cfg
    ):
        super().__init__()
        # self.model=testNet(cfg)
        self.model=ISINS(cfg)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)
        self.lr=cfg.train.learning_rate
        self.train_batch_size=cfg.train.batch_size
        self.eval_batch_size=cfg.evaluation.batch_size
        self.automatic_optimization=False
        self.max_epoch=cfg.train.max_epoch
        self.level=cfg.model.level
        self.resolution=cfg.model.resolution
        self.cfg=cfg
        self.smooth_it=cfg.model.smooth
        self.hooks=[]

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()
        self.trainer.model.train()
        self.model.train()
        # for name, module in self.model.named_children():
        #     hook = module.register_forward_hook(
        #         lambda m, inp, out, name=name: self._forward_hook(m, inp, out, name)
        #     )
        #     self.hooks.append(hook)


    def assembly_K_F_coo(self, batch):
        level_coo=calculate_legal_coordinates(batch["coordinates"].long(),self.level)
        K_value,f_value=node_KF_assembly(batch_id=batch["coordinates"][:,0],
                                        nodetype=batch["node_type"],
                                        K_node={"hard":batch["K_node"].view(self.train_batch_size,256,-1),"soft":batch["K_node_soft"].view(self.train_batch_size,256,-1)},
                                        F_node={"hard":batch["F_node"].view(self.train_batch_size,256,-1),"soft":batch["F_node_soft"].view(self.train_batch_size,256,-1)},
                            )

        node_stiffness=ME.SparseTensor(coordinates=batch["coordinates"].float(), features=K_value)
        K_list=calculate_level_K_list(node_stiffness,level_coo,self.train_batch_size,self.resolution,self.level)
        
        f_batch_list=[f_value]
        for i in range (1,self.level):
            f_batch_list.append(fine2coarse_interpolation(ME.SparseTensor(coordinates=level_coo[i-1].float(),features=f_batch_list[i-1]),
                                                            level_coo[i],self.resolution/(2**i)))
        return K_list,f_batch_list,level_coo

    def training_step(self, batch, batch_idx):
        self.model.train()
        K_list,f_batch_list,level_coo=self.assembly_K_F_coo(batch)
        torch.cuda.empty_cache()
        U_list=self.model(batch,level_coo,K_list,f_batch_list,self.smooth_it,self.train_batch_size)

        # for name, param in self.model.named_parameters():
        #     status = "✅ 有梯度" if param.grad is not None else "❌ 无梯度"
        #     print(f"{name.ljust(30)} {status}")
        torch.cuda.empty_cache()
        weight=0.1*(self.trainer.estimated_stepping_batches-self.global_step)/self.trainer.estimated_stepping_batches
        optimizer = self.optimizers()
        # (optimizer,K_list,U_list,F_list,coo_list,batch_size,weight)
        loss=MPE_loss_all_layer(optimizer,K_list,U_list,f_batch_list,level_coo,self.train_batch_size,weight)
        
        self.log('lr', optimizer.state_dict()['param_groups'][0]['lr'],prog_bar=False,sync_dist=True,batch_size=self.train_batch_size)
        self.log('U-weight', weight,prog_bar=False,sync_dist=True,batch_size=self.train_batch_size)
        for key,value in loss.items():
            if key=='loss':
                self.log(key, value,prog_bar=True,batch_size=self.train_batch_size)
            else :
                self.log(key, value,prog_bar=False,batch_size=self.train_batch_size)
        self.ema.update(self.model.parameters())
        if self.global_step % 8 == 0:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            self.ema.restore(self.model.parameters())

        sch = self.lr_schedulers()
        self.log('lr', sch.get_lr()[0],prog_bar=False,sync_dist=True)
        sch.step()

        


    def on_train_epoch_end(self):
        

        # 确认优化器包含所有参数
        for hook in self.hooks:
            hook.remove()
        if self.global_rank==0:
            print('Epoch-',self.current_epoch,'- Finish.')
        
    
    def on_validation_model_eval(self) -> None:
        self.trainer.model.train()
        self.model.train()
        torch.cuda.empty_cache()
        
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        K_list,f_batch_list,level_coo=self.assembly_K_F_coo(batch)
        validate_dict=validate(self.model,K_list,f_batch_list,level_coo,batch,self.eval_batch_size,self.level,self.smooth_it,self.resolution)
        for key,value in validate_dict.items():
            if key=='error' or key=='loss':
                self.log(key, value,prog_bar=True,batch_size=self.train_batch_size, sync_dist=True)
            else :
                self.log(key, value,prog_bar=False,batch_size=self.train_batch_size, sync_dist=True)
        # self.log_dict(validate_dict,logger =True,sync_dist=True,batch_size=self.eval_batch_size)

    def on_test_model_eval(self) -> None:
        self.trainer.model.train()
        self.model.train()
        torch.cuda.empty_cache()
    def test_step(self, batch, batch_idx):
        self.trainer.model.train()
        self.model.train()
        torch.cuda.empty_cache()
        validate_dict=validate(self.model,batch,self.eval_batch_size,self.resolution)
        for key,value in validate_dict.items():
            if key=='error' or key=='loss':
                self.log(key, value,prog_bar=True,batch_size=self.train_batch_size, sync_dist=True)
            else :
                self.log(key, value,prog_bar=False,batch_size=self.train_batch_size, sync_dist=True)
        
        
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.model.parameters(),self.lr,weight_decay=0.01)
        lr_schedulers=CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=20,
            max_lr=self.lr,
            min_lr=1e-5,
            gamma=0.5,
        )
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_schedulers
            }

    def _forward_hook(self, module, inp, output, name):
        # 打印模块名、输入输出的梯度信息
        print(f"模块名: {name}")
        print(f"  输入 grad_fn: {inp[0].grad_fn if isinstance(inp[0], torch.Tensor) else 'None'}")
        print(f"  输出 grad_fn: {output.grad_fn if isinstance(output, torch.Tensor) else 'None'}")
        print("---")

        