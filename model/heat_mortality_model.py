import torch
import torch.nn as nn
import os
import lightning.pytorch as pl
from .utils import population_preprocessing, weather_preprocessing
from .loss_function import HeatMortalityLoss
from .metric import HeatMortalityAccuracy
import pandas as pd
import numpy as np


class HeatMortalityBase(pl.LightningModule):
    
    '''
        Base class for heat mortality modelling. The main part is to process the data for training.
        The implementation of the network is done in the children classes.
        
        in_channel: depend on temperature feature (1 if only mean/min/max is used, 3 if these are used together)
        out_channel: number of deaths predicted, in Germany example this is 30 (15 age groups, 2 gender)
    '''
    
    def __init__(self, population_dir, weather_dir, in_channel=1, out_channel=30, kernel_days=10, 
                val_split = 0.2, n_layers = 3, hidden_dim_init=128,
                useExp=False, activation=False, population_interpolation=True, baseline_mode='bottom_10',
                weather_data = 'de_1km', t_type = 'mean', weekday_correction=False):
        super().__init__()
        self.val_split = val_split

        self.death_pred = None
        self.death_pred_week = None
        
        valid_t_type = {'max', 'mean', 'min', 'eval'}
        valid_baseline_modes = {'constant', 'moving_avg', 'bottom_10'}
        valid_weather_data = {'CERRA', 'de_1km'}
        assert baseline_mode in valid_baseline_modes
        assert weather_data in valid_weather_data
        assert t_type in valid_t_type
        self.t_type = t_type
        weather_points = {'CERRA':4, 'de_1km': 1}
        self.weather_data = weather_data
        self.points_per_day = weather_points[weather_data]
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_days = kernel_days
        self.kernel_size = self.kernel_days * self.points_per_day
        self.stride = self.points_per_day
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim_init
        self.useExp = useExp
        
        # Get the data for the model
        self._get_weather_data(weather_dir)
        self.population_interpolation = population_interpolation
        self.baseline_mode = baseline_mode
        self._init_data(population_dir)
        
        self.activation = activation
        self._init_model()      

        # parameters for day-of-the-week correction, only 6 parameter used because the first day is considered as standard
        if weekday_correction:
            self.weekday_correction = nn.Parameter(torch.ones([6], dtype=torch.float32))
        else:
            self.weekday_correction = torch.ones([6], dtype=torch.float32)
        
        self.loss = HeatMortalityLoss()
        self.metric = HeatMortalityAccuracy()
        
        self.save_hyperparameters()
        
    def forward(self, x):
        raise NotImplementedError
    
    def training_step(self, batch, batch_index):
        return self._step(batch, batch_index, mode='train')
    
    def validation_step(self, batch, batch_index):
        self._step(batch, batch_index, mode='val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, threshold=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'train_loss',
            }}
    
    def _init_model(self):
        raise NotImplementedError
         
    def _init_data(self, population_dir):
        # preparing population and death data
        preprocessed = population_preprocessing(population_dir, 
                                                kernel_days=self.kernel_days, 
                                                val_split=self.val_split,
                                                baseline_mode=self.baseline_mode)
        self.population, self.deaths, self.train_labels, self.val_labels, self.train_weeks, self.skip_days, self.basic_death_rate = preprocessed
        self.basic_death_case = self.basic_death_rate * self.population
    
    def on_train_start(self):
        self.tensor_t = self.tensor_t.to(device=self.device)
        self.train_labels = [t.to(self.device) for t in self.train_labels]
        self.val_labels = [t.to(self.device) for t in self.val_labels]
        self.basic_death_case = self.basic_death_case.to(self.device)
        self.weekday_correction = self.weekday_correction.to(self.device)
        
    def _step(self, batch, batch_index, mode='train'):
        if mode == 'train':
            # arr_weather = torch.cat([temperature, humidity], axis = 1)
            # Add a random noise to the input data
            rand_t = torch.rand_like(self.tensor_t, device=self.device) - 0.5
            arr_weather = (self.tensor_t + rand_t) / 15
            outputs = self.forward(arr_weather)
            
            # Save the prediction in a variable, which is later used for calculating train loss and validation loss
            self.death_pred = self._death_pred(outputs)
            self.death_pred_week = self.death_pred[self.skip_days[0]:-self.skip_days[1]]
            n_days, n_kreis, n_age_group, n_sex = self.death_pred_week.shape
            self.death_pred_week = self.death_pred_week.reshape(n_days//7, 7, n_kreis, n_age_group, n_sex).sum(1)
            
            # Fit the output of training set to a size suitable for loss calculation
            death_pred = self.death_pred[:7 * self.train_weeks + self.skip_days[0]]
            death_pred_week = self.death_pred_week[:self.train_weeks]
            label = self.train_labels

        else:
            # Fit the output of validation set to a size suitable for loss calculation
            death_pred = self.death_pred[7 * self.train_weeks + self.skip_days[0]:]
            death_pred_week = self.death_pred_week[self.train_weeks:]
            label = self.val_labels
        
        loss = self.loss(death_pred, death_pred_week, *label)
        accuracy = self.metric(death_pred, death_pred_week, *label[:2])
        
        if mode == 'train':
            self.log_dict({f'{mode}_loss': loss}, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log_dict({f'{mode}_loss': loss, 
                            f'{mode}_daily_mse': accuracy[0], 
                            f'{mode}_weekly_mse': accuracy[1]},
                            on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def _death_pred(self, outputs):
        weekday_corr = torch.cat([torch.ones([1], dtype=torch.float32, device=self.device), self.weekday_correction.to(self.device)])
        corr = weekday_corr.repeat(outputs.shape[0] // 7 + 1).view(-1,1,1,1)[:outputs.shape[0]]
        prediction = (0.99 + outputs.clip(-0.99) + 0.01 * (outputs.clip(max=-0.99) + 0.99).exp()) * \
                      corr * self.basic_death_case[self.kernel_days-1:]
        return prediction
    
    def _get_weather_data(self, weather_dir):
        self.tensor_t = weather_preprocessing(weather_dir, self.weather_data, self.in_channel, self.t_type)
        
        
class HeatMortality_FC_RELU(HeatMortalityBase):
    
    '''
        Fully connected model for death case prediction
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.useExp
        
    def _init_model(self):
        self.conv1 = nn.Conv1d(in_channels=self.in_channel, out_channels=self.hidden_dim, kernel_size=self.kernel_size, stride=self.stride)
        if not self.activation:
            self.activation = nn.ReLU()
        layers = []
        for i in range(self.n_layers):
            if i == 1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(self.activation)
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
                layers.append(self.activation)
                self.hidden_dim //= 2                
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(self.hidden_dim, self.out_channel)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        res = self.conv1(x).permute([2, 0, 1])
        res = self.layers(res)
        res = self.fc(res)
        if self.out_channel > 1:
            res = res.reshape(*res.shape[:-1], -1, 2)
        return res

class Conv1dPos(nn.Conv1d):
    '''
        1d Convolution layer with non negative weights (bias can be negative)
    '''
    def forward(self, input):
        return self._conv_forward(input, self.weight**2, self.bias)
    
class LinearPos(nn.Linear):
    '''
        Fully connected layer with non negative weights (bias can be negative)
    '''
    def forward(self, input):
        return nn.functional.linear(input, self.weight**2, self.bias)

class HeatMortality_EXP(HeatMortalityBase):
    
    '''
        Exponential model for death case prediction
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _init_model(self):
        self.multi_dim = 32
        self.norm = nn.BatchNorm3d(1, track_running_stats=True)
        self.multiply = nn.Conv1d(in_channels=self.in_channel, 
                                    out_channels=self.multi_dim, 
                                    kernel_size=1, 
                                    stride=1)
        self.conv1 = nn.Conv1d(in_channels=self.in_channel, 
                               out_channels=self.hidden_dim, 
                               kernel_size=self.kernel_size, 
                               stride=self.stride)
        self.conv2 = Conv1dPos(in_channels=self.multi_dim, 
                               out_channels=self.hidden_dim, 
                               kernel_size=self.kernel_size, 
                               stride=self.stride)
        self.fc = LinearPos(self.hidden_dim, self.out_channel)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.norm(x).squeeze(0).squeeze(0)
        x_sum = self.conv1(x).permute([2, 0, 1])
        x_sum_exp = x_sum.exp()
        x_exp = self.multiply(x).exp()
        x_exp_sum = self.conv2(x_exp).permute([2, 0, 1])
    
        res = x_exp_sum + x_sum_exp
        res = self.fc(res)
        if self.out_channel > 1:
            res = res.reshape(*res.shape[:-1], -1, 2)
        return res