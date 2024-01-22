#!/usr/bin/env python

import torch
import torch.nn as nn
import model.dataset as dataset
import model.heat_mortality_model as heat_mortality_model
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, EarlyStopping
import model.config as config
import lightning.pytorch as pl


def main(kernel_days=6, useExp=True, activation=None, version=1, t_type='mean', weather_data='de_1km',
         baseline='bottom_10', weekday_correction=True):
    if not useExp:
        activation = True
    file_name = 'exp_' * useExp + f'kernel_{kernel_days}' + weather_data + '_' + t_type + '_' + baseline
    config.to_hpc_copy()

    train_data = dataset.DummyDataset(10)
    val_data = dataset.DummyDataset(1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=None, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=None, num_workers=0)
    
    if activation is True:
        activation = nn.ReLU()
        
    # Hyperparameters for the model
    params = {'kernel_days':kernel_days,
              'useExp': useExp,
              'population_interpolation':True, 
              'baseline_mode':baseline, 
              'weather_data': weather_data,
              "t_type": t_type,
              'in_channel': 1, 
              'weekday_correction': weekday_correction}

    if not useExp:
        lit_model = heat_mortality_model.HeatMortality_FC_RELU(config.POPULATION_DIR, 
                                                                config.WEATHER_DIR,
                                                                **params)
    else:
        lit_model = heat_mortality_model.HeatMortality_EXP(config.POPULATION_DIR, 
                                                                config.WEATHER_DIR, 
                                                                **params)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="lightning_logs/" + file_name, 
                                        filename=f'v{version}-' + '{epoch}-{train_loss:.2E}-{val_loss:.2E}',
                                        save_weights_only=True)
    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=50)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs/" + file_name, name='v' + str(version))
    trainer = pl.Trainer(   max_epochs=20000, 
                            enable_progress_bar = False,
                            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback], 
                            logger=tb_logger,
                            default_root_dir="./stat_dict/",
                            precision=64,
                            gradient_clip_val=1,
                            num_sanity_val_steps=0)
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
    

