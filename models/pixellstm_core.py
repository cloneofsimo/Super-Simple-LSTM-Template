from omegaconf.dictconfig import DictConfig
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils

from random import randint


class pixellstm(nn.Module):
    """
    Pixel LSTM by Simo Ryu, implemented for simplicity & hackability.
    """
    def __init__(self, emb = 64, n_channels = 1, img_size = 28, n_enc_layer = 3, n_dec_layer = 3, n_class = 10):
        super(pixellstm, self).__init__()
        self.class_emb = nn.Embedding(n_class, emb)
        self.n_class = n_class
        self.device = "cuda:0"
        self.emb = emb
        self.n_channels = n_channels
        self.img_size = img_size
        self.encoder = nn.Sequential(
            nn.Linear(n_channels, emb),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(emb, n_channels),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(input_size = emb, hidden_size = emb, batch_first = True, num_layers = 8)
        self.loss = nn.MSELoss()
    
    def main(self, x, y): #x is B, L, C
        B, L, C = x.size()
        x = self.encoder(x)
        y_in = self.class_emb(y).reshape(B, 1, self.emb)
        x_in = torch.cat([y_in, x], dim = 1)
        x_out_pred = self.decoder(self.lstm(x_in)[0])

        return x_out_pred[:, -1:, :]
    
    def forward(self, x, y, inference = False):
        """
        x should have dim B, C, H, W
        """
        B, C, H, W = x.size()
        x_ = x.reshape(B, C, -1).transpose(1, 2) # B, H*W, C
        x = self.encoder(x_) # B, H*W, emb
        y_in = self.class_emb(y).reshape(B, 1, self.emb)
        x_in = torch.cat([y_in, x[:, :-1, :]], dim = 1)

        x_out_pred = self.decoder(self.lstm(x_in)[0])
        if inference:
            return x_out_pred
        else:
            loss = self.loss(x_, x_out_pred)
            return loss 

    def sample(self):
        y_in = torch.arange(self.n_class).reshape(self.n_class, 1).to(self.device)
        x = torch.zeros(self.n_class, 0, self.n_channels).to(self.device)
        for _ in range(self.img_size * self.img_size):
            x_out = self.main(x, y_in)
            x = torch.cat([x, x_out], dim = 1)
        
        
        # B, H*H, C
        x = x.transpose(1, 2).reshape(self.n_class, self.n_channels, self.img_size, self.img_size)
        print(x.size())
        return x
        
        