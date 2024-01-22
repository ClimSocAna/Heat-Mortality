import torch
from torch import nn
from torchmetrics import Metric
from . import config
 
class HeatMortalityAccuracy(Metric):
    
    '''
        Metric for calculating the daily and weekly mse loss for death cases in Germany
    '''
    
    def __init__(self):
        super().__init__()
        self.add_state("mse_daily", default=torch.tensor(0, dtype=torch.double))
        self.add_state("mse_weekly", default=torch.tensor(0, dtype=torch.double))
    
    def update(self, death_pred, death_pred_week, death_daily, death_de):
        death_pred = death_pred.sum([1,2,3])
        death_pred_week = death_pred_week.sum([1,2,3])
        death_true_daily = death_daily.sum(1)
        death_true_weekly = death_de.sum([1,2])

        self.mse_daily += nn.MSELoss()(death_pred.detach(), death_true_daily)
        self.mse_weekly += nn.MSELoss()(death_pred_week.detach(), death_true_weekly)

    def compute(self):
        return self.mse_daily, self.mse_weekly