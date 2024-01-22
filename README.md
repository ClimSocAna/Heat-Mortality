# Heatrelatedt Mortality Estimation

Shortcut: [Remarks](https://github.com/ClimSocAna/Heat-Mortality#Remarks) - [Datasets](https://github.com/ClimSocAna/Heat-Mortality#Datasets)

## Remarks

This repository contains code of shallow neural network to estimate the mortality in Germany. The estimated mortality can than be used to calculate heat-related mortality.

There are two seperate models used in the code:
- Model to estimate mortality based on the temperature (MortModel)
- Model to map weather station data to district level temperature (TempModel)

This work improves the estimation in the following ways:
- Local risk estimation (district level)
- Daily risk estimation (illustration of lag effect)

The master branch will not be updated after the publication of the paper.

## Datasets

The following data were used to train the model.

Temperature
- [CERRA reanalysis data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-cerra-single-levels?tab=overview) (Copernicus, open data)
- [1x1 km daily temperature data](https://doi.org/10.1016/j.envres.2022.115062) (Helmholz Munich, non-open data)
- [Deutscher Wetterdienst(DWD) station data](https://www.dwd.de/DE/klimaumwelt/cdc/cdc_node.html) (Climate Data Center, open data)
- [ERA5 daily statistics calculator](https://cds.climate.copernicus.eu/cdsapp#!/software/app-c3s-daily-era5-statistics?tab=overview) (Copernicus, open data)
- [Climate projection](https://aims2.llnl.gov/search) (EC-Earth3, open data)

[District level population](https://www.regionalstatistik.de/genesis/online?operation=statistic&levelindex=0&levelid=1705590504131&code=12411#abreadcrumb) (Statistische Ämter des Bundes und der Länder, open data)

[Death statistic](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Sterbefaelle-Lebenserwartung/Tabellen/sonderauswertung-sterbefaelle.html) (DESTATIS, open data)

[Coordinate of each district](https://public.opendatasoft.com/explore/dataset/georef-germany-kreis/information/?disjunctive.lan_code&disjunctive.lan_name&disjunctive.krs_code&disjunctive.krs_name&disjunctive.krs_name_short&sort=year&location=6,51.32946,10.45403&basemap=jawg.light) (open data)

The population and death data was preprocessed and is avaiable in data folder. The district level temperature data from 2021 to 2023 (acquired with TempModel based on DWD data) is avaiable for testing.

The training data is not available due to the data agreement. Instead, we provide the trained parameters in params folder.

The climate projection data is not available due to the size.

## Data pipeline

The code is mainly for data exploration and not suitable for monitoring purpose. The model itself contains many raw data for examinations. A proper data pipeline needs to be implemented for setting up surveillance system.

## Model and training

The 
