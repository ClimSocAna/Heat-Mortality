import torch
import numpy as np
import lightning.pytorch as pl
from .utils import age_grouping, land_grouping
from . import config
 
class HeatMortalityLoss(pl.LightningModule):
    
    '''
        Poisson loss for calculated the loss between the output (death cases for each district, age group, sex) and target.
        The grouping of the death cases is implemented in a seperate method.
    '''
    
    def __init__(self):
        super().__init__()
        # number of districts in each Land

        self.loss = torch.nn.PoissonNLLLoss(log_input=False, full=True)

    def forward(self, death_pred, death_pred_week, death_daily, death_de, death_land):
        
        # loss for the daily death in each Bundesland
        death_pred_daily_land_age_sex = land_grouping(death_pred)
        death_land_daily = death_pred_daily_land_age_sex.sum(dim=[-1,-2])
        loss_daily = self.loss(death_land_daily, death_daily)

        # raw prediction of weekly death in detailed age group in each Bundesland, age group and sex
        death_pred_weekly_land_age_sex = land_grouping(death_pred_week)
        
        # loss for the weekly death in detailed age group in Germany
        death_pred_weekly_de = death_pred_weekly_land_age_sex.sum(dim=1)
        loss_weekly_de = self.loss(death_pred_weekly_de, death_de)

        # loss for the weekly death in age group in each Bundesland
        death_pred_weekly_land = age_grouping(death_pred_weekly_land_age_sex, 2, 
                                            [list(range(8)), [8, 9], [10, 11], [12, 13, 14]])
        loss_weekly_land = self.loss(death_pred_weekly_land, death_land)
    
        return loss_daily + loss_weekly_de + loss_weekly_land
    
