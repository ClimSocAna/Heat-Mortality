import os, sys

import torch
import torch.nn as nn
from .dataset import DummyDataset

import config
import lightning.pytorch as pl
import pandas as pd
import numpy as np



class TempModel(pl.LightningModule):
    '''
        Attention model for district level temperature estimation.

        meaasure:       the first two columns are the coordinate of the point, other colomns are values of the measurement / projection
        target:         dataset of the district level temperature used for training and validation
        geo_district:   coordinates of each district, used to initiate the coordinates in the attention model
    '''
    def __init__(self, measure, target, geo_district, val_split = 0.2):
        super().__init__()
        self.target = target
        self.train_size = int(self.target.shape[0] * (1 - val_split))
        self.geo_station = torch.tensor(measure.iloc[:, [0,1]].values) 
        self.geo_kreis = nn.Parameter(torch.tensor(geo_district, requires_grad=True)) 
        self.t_measurement = torch.tensor(measure.iloc[:, 2:].values)
        self.scale = nn.Parameter(torch.ones([400, 1, 1], dtype=torch.float32)) # scaling factor of the attention model
        self.bias = nn.Parameter(torch.ones([400, 1], dtype=torch.float32))
        self.mask = self.t_measurement < -100
        
    def forward(self, x):
        mask = x < -100
        # calculate the attention value
        v = (-(self.scale.abs()) * ((torch.cdist(self.geo_kreis, self.geo_station).unsqueeze(-1) * ~mask) + 100 * mask)).exp()
        # normalize the sum of attention to 1
        v = ((v / v.sum(dim=1, keepdim=True)) * x).sum(1) + self.bias
        return v.T
        
    def training_step(self, batch, batch_index):
        return self._step(batch, batch_index, mode='train')
    
    def validation_step(self, batch, batch_index):
        self._step(batch, batch_index, mode='val')
    
    def on_train_start(self):
        self.t_measurement = self.t_measurement.to(device=self.device)
        self.target = self.target.to(device=self.device)
        self.geo_station = self.geo_station.to(device=self.device)
        self.geo_kreis = self.geo_kreis.to(device=self.device)
        
    def _step(self, batch, batch_index, mode='train'):
        
        batch_size = 1000
        n_batches = 2
        
        if mode == 'train':
            self.pred = self.forward(self.t_measurement[:, batch_size * (batch_index % n_batches):batch_size * (batch_index % n_batches + 1)])
            loss = nn.MSELoss()(self.target[:batch_size * (batch_index % n_batches):batch_size * (batch_index % n_batches + 1)], self.pred)
        else:
            self.pred = self.forward(self.t_measurement[:, batch_size * n_batches:])
            loss = nn.MSELoss()(self.target[batch_size * batch_index:], self.pred)
        self.log_dict({f'{mode}_loss': loss}, on_step=True, on_epoch=True, prog_bar=True)

        return loss

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, threshold=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'train_loss',
            }}
        

