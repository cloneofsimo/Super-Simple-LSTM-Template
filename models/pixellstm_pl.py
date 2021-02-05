from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils

from .pixellstm_core import pixellstm

def show_img(img):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Sampled Imgs")
    plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


class pixellstm_pl(pl.LightningModule):
    def __init__(self, cfg : DictConfig):
        super().__init__()
        self.cfg = cfg
        self.core = pixellstm()
    
    def forward(self, x, y):
        return self.core(x, y) #returns loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.forward(x, y)
    
        tensorboard_log = {
                'train_loss': loss
        }

        return {'loss': loss, 'log': tensorboard_log}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.forward(x, y)
    
        tensorboard_log = {
                'val_loss': loss
        }

        return {'val_loss': loss, 'log': tensorboard_log}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.sampleandshow()
        tensorboard_log = {
            'val_loss': loss
        }
        return {'val_loss': loss, 'log': tensorboard_log}

    


    def configure_optimizers(self):
        if self.cfg.train.optim == 'adam':
            return optim.Adam(
                self.parameters(),
                lr=self.cfg.train.lr,
                betas=self.cfg.train.betas,
                weight_decay= self.cfg.train.weight_decay,
                amsgrad=True
            )
        else:
            raise NotImplementedError
    
    def sampleandshow(self):
        img = self.core.sample()
        show_img(img)

