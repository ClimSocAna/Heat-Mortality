import torch
import os
import numpy as np
import pandas as pd
from datetime import date

def age_grouping(data, axis, groups):

    new_data = []
    for g in groups:
        g = torch.tensor(g, device=data.device)
        new_data.append(torch.index_select(data, axis, g).sum(dim=axis))
    new_data = torch.stack(new_data, dim=axis)
    return new_data

_kreis_count = torch.Tensor([0, 15, 1, 45, 2, 53, 26, 36, 44, 96, 6, 1, 18, 8, 13, 14, 22])
_kreis_indexing = _kreis_count.cumsum(0, dtype=torch.int)

def land_grouping(x, land_axis = 1):
    arr = []
    for i in range(len(_kreis_count) - 1):
        arr.append(x.index_select(land_axis, torch.arange(_kreis_indexing[i], _kreis_indexing[i+1], device=x.device)).sum(dim=land_axis))
    arr = torch.stack(arr, dim = land_axis)
    return arr

def population_preprocessing(population_dir, kernel_days, val_split, baseline_mode, interpolation=False, begin=2011, end=2020):
    
    df_pop = pd.read_csv(os.path.join(population_dir, 'population/population_ab2011.csv'))
    arr_pop = df_pop.iloc[:-8000, 5:].values.reshape(10, 400, 20, -1)
    pop_interpolation = [arr_pop[0]] * 365
    for year in range(2012, 2021):
        days = 365 + (year % 4 == 0)
        for i in range(1, days + 1):
            if interpolation:
                pop_interpolation.append((arr_pop[year-2011] * i + arr_pop[year-2012] * (days - i)) // days)
            else:
                pop_interpolation.append(arr_pop[year-2011])
    arr_pop = np.stack(pop_interpolation, axis=0)
    population = age_grouping(torch.tensor(arr_pop), 2, [list(range(6))] + [[i] for i in range(6, 20)])
    
    death_daily = torch.tensor(np.load(os.path.join(population_dir, 'death_cases/de_daily.npy')))
    death_de_men = torch.tensor(np.load(os.path.join(population_dir, 'death_cases/men_de_age_week.npy')))
    death_de_women = torch.tensor(np.load(os.path.join(population_dir, 'death_cases/women_de_age_week.npy')))
    death_land_men = torch.tensor(np.load(os.path.join(population_dir, 'death_cases/men_land_age_week.npy')))
    death_land_women = torch.tensor(np.load(os.path.join(population_dir, 'death_cases/women_land_age_week.npy')))
    
    death_daily = death_daily[0, :, 11*366 : 21*366]
    death_daily = death_daily[death_daily != -1].reshape(16, -1).swapaxes(0, 1)
    
    death_de_men = death_de_men[0, 1:, 11*53 : 21*53]
    death_de_women = death_de_women[0, 1:, 11*53 : 21*53]
    death_de_men = death_de_men[death_de_men != -1].reshape(15, -1)
    death_de_women = death_de_women[death_de_women != -1].reshape(15, -1)
    death_de = torch.stack([death_de_men, death_de_women], dim = -1).swapaxes(0,1)
    
    death_land_men = death_land_men[:, 1:, 11*53 : 21*53]
    death_land_women = death_land_women[:, 1:, 11*53 : 21*53]
    death_land_men = death_land_men[death_land_men != -1].reshape(16, 4, -1)
    death_land_women = death_land_women[death_land_women != -1].reshape(16, 4, -1)
    death_land = torch.stack([death_land_men, death_land_women], dim = -1).moveaxis(2,0)
    death_statistic = [death_daily, death_de, death_land]
    
    begin_weekday = date(begin, 1, 1).weekday()
    end_weekday = date(end, 12, 31).weekday()
    skip_days_begin = (1 - begin_weekday - kernel_days) % 7
    skip_days_end = (end_weekday + 1) % 7
    skip_weeks_begin = (begin_weekday + kernel_days - 2) // 7 + (begin_weekday < 4)
    skip_weeks_end = int(3 <= end_weekday <= 5)
    train_weeks = int((death_de.shape[0] - skip_weeks_begin - skip_weeks_end) * (1 - val_split))
    skip_days = skip_days_begin, skip_days_end
    
    train_death_daily = death_daily[kernel_days-1:7 * train_weeks + skip_days_begin + kernel_days-1]
    train_death_de = death_de[skip_weeks_begin:skip_weeks_begin+train_weeks]
    train_death_land = death_land[skip_weeks_begin:skip_weeks_begin+train_weeks]
    train_labels = [train_death_daily, train_death_de, train_death_land]

    val_death_daily = death_daily[7 * train_weeks + skip_days_begin + kernel_days - 1:]
    val_death_de = death_de[skip_weeks_begin+train_weeks:-skip_weeks_end]
    val_death_land = death_land[skip_weeks_begin+train_weeks:-skip_weeks_end]
    val_labels = [val_death_daily, val_death_de, val_death_land]
    
    if baseline_mode == 'constant':
        base_death_rate = death_de.sum(0) / death_de.shape[0] / 7 / population.sum(1).float().mean(0)
        base_death_rate = base_death_rate.repeat(3653,1,1,1)
        
    death = torch.cat([death_de[:51], death_de], dim=0)
    arr = []
    for i in range(death.shape[0] - 51):
        if baseline_mode == 'moving_avg':
            arr.append(death[i:i+52].sum(0) / 364)
        elif baseline_mode == 'bottom_10':
            arr.append(death[i:i+52].topk(10, dim=0, largest=False).values.sum([0]) / 70)
    death = torch.stack(arr).repeat_interleave(7, 0)
    base_death_rate = (death[:population.shape[0]] / population.sum(1)).unsqueeze(1)

    return population, death_statistic, train_labels, val_labels, train_weeks, skip_days, base_death_rate

def weather_preprocessing(weather_dir, data_source, in_channel=1, t_type='mean'):

    if t_type == 'eval':
        return torch.tensor(0)
    
    df_t= pd.read_csv(os.path.join(weather_dir, 'CERRA/kreis_temperature_2011-2020.csv'), index_col=0)
    cerra_t = torch.tensor(df_t.values).T - 273.15
      
    def fill_t_na(t_type):
        # used to fill the na value in the Helmholz Munich data
        regional = pd.read_csv(os.path.join(weather_dir,f'de_1km/kreis_T{t_type}_2000-2021.csv'), index_col=0).to_numpy()
        select = []
        for i in range(11, 21):
            select += [366* i + j for j in range(0, 366) if not (j == 59 and i % 4 != 0)]
        regional_t = regional[select].T
        cerra_mean = cerra_t.reshape(400, -1, 4).mean(-1)
        diff = (cerra_mean - regional_t).nanmean(1)
        for i in range(regional_t.shape[0]):
            for j in range(regional_t.shape[1]):
                if np.isnan(regional_t[i, j]):
                    regional_t[i, j] = cerra_mean[i, j] - diff[i]
        return torch.tensor(regional_t)
    
    if data_source == "de_1km":    
        if in_channel == 1:
            tensor_t = fill_t_na(t_type)
        elif in_channel == 3:
            tensor_t = torch.stack([fill_t_na(t_type) for t_type in ['mean', 'max', 'min']], dim=1)
    else:
        tensor_t = cerra_t
        
    return tensor_t
