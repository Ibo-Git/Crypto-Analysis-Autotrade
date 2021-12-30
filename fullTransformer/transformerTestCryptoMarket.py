import math
import os
import pickle
import warnings
from functools import reduce
from operator import itemgetter

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader

from customDataset import CustomDataset
from transformerModel import Trainer, TransformerModel


def load_asset(assetname, num_intervals):
    # load data
    filename = f'{assetname}_1.csv'
    df= pd.read_csv(
        os.path.join('datasets', 'assets', filename), 
        names=["Datetime", "Open", "High", "Low", "Close", "Volume", "Number of trades"]
    )

    # get number of intervals e.g. for 5m and num_intervals of 17280 you get 60 days of data
    df = df.tail(num_intervals)
    return df



def custom_candles(dataframe, asset_interval):
    # create list with each dataframe row in a sublist
    data = dataframe.drop(['Datetime'], axis=1).to_numpy().tolist()

    # concatenate 1 minutes to length of asset_interval and separate overlapping intervals
    data_separated_by_interval = [[data[i + n:i + n + asset_interval] for n in range(0, len(data) - asset_interval, asset_interval)] for i in range(asset_interval)]

    # combine the 1 minute candles for each feature to one candle of asset_interval length
    for num_interval, interval in enumerate(data_separated_by_interval):
        for num_candle, candles in enumerate(interval):
            candles = list(zip(*candles))
            data_separated_by_interval[num_interval][num_candle] = [
                candles[0][0], # open
                max(candles[1]), # high
                min(candles[2]), # low
                candles[3][-1], # close
                sum(candles[4]), # volume
                sum(candles[5]) # number of trades
            ]

    # add additional features e.g. RSI
    rsi_window = 10

    for num_interval, interval in enumerate(data_separated_by_interval):
        # define temporary dataframe to determine additional features using pandas_ta
        df_temp = pd.DataFrame(interval, columns=["Open", "High", "Low", "Close", "Volume", "Number of trades"])
        df_temp.ta.rsi(close='Close', length=rsi_window, append=True)
        df_temp = df_temp.tail(len(df_temp) - rsi_window) # remove NaN entries 
        data_temp = [] # temporary list to convert dataframe to list

        for _, row in df_temp.iterrows():
            # add more features if needed
            data_temp.append([
                row['Open'], 
                row['High'], 
                row['Low'], 
                row['Close'],
                row['Volume'],
                row['Number of trades'],
                row['RSI_10'],
                1 if (row['High'] / row['Open']) - 1 >= 0.006 else 0,
                1 if (row['High'] / row['Open']) - 1 <  0.006 else 0,
            ])
        
        data_separated_by_interval[num_interval] = data_temp

    return data_separated_by_interval


def data_preprocessing(params, assets, features):
    data = {}
    # load each asset using custom candles
    if params.overwrite_saved_data or not os.path.isfile('data.pkl'):
        for asset_key, asset in assets.items():
            #crypto_df = yf.download(tickers=asset['api-name'], period=asset['period'], interval=asset['interval'])
            crypto_df = load_asset(asset['api-name'], num_intervals=asset['num_intervals'])
            data[asset_key] = custom_candles(crypto_df, params.asset_interval)

        with open('data.pkl', 'wb') as file:
            pickle.dump(data, file)
            
    else:
        with open('data.pkl', 'rb') as file:
            data = pickle.load(file)


    # split into training and validation sequences
    train_sequences = {}
    val_sequences = {}

    for asset_key, asset in data.items():
        train_sequences[asset_key]  = []
        val_sequences[asset_key]  = []
        interval_avg_train = []
        interval_avg_val = []

        for num_interval, interval in enumerate(asset):
            train_sequences[asset_key].append(interval[0:math.floor(len(interval) * params.split_percent)])
            val_sequences[asset_key].append(interval[math.floor(len(interval) * params.split_percent):])

            # determine average change for each interval
            interval_avg_train.append(np.mean(np.abs(np.diff(list(zip(*train_sequences[asset_key][num_interval])))), 1))
            interval_avg_val.append(np.mean(np.abs(np.diff(list(zip(*val_sequences[asset_key][num_interval])))), 1))

        # average the avg change over all intervals
        feature_avg = {
            'train': {asset_key: np.mean(list(zip(*interval_avg_train)), 1)},
            'val': {asset_key: np.mean(list(zip(*interval_avg_val)), 1)}
        }

    # scaling training and validation sequences according to their scaling-mode
    scale_values = {}
    features_len = len(features.values())

    for asset in data:
        # determine min and max values 
        scale_values[asset] = {
            'min': [
                min(data[asset], key=itemgetter(n))[n] if list(features.values())[n]['scaling-mode'] == 'min-max-scaler' 
                else list(features.values())[n]['scaling-limits'][0] if list(features.values())[n]['scaling-mode'] == 'limit-scaler'
                else warnings.warn("Undefined scale mode!")
                for n in range(features_len)
            ],
            'max': [
                max(data[asset], key=itemgetter(n))[n] if list(features.values())[n]['scaling-mode'] == 'min-max-scaler' 
                else list(features.values())[n]['scaling-limits'][1] if list(features.values())[n]['scaling-mode'] == 'limit-scaler'
                else warnings.warn("Undefined scale mode!")
                for n in range(features_len)
            ],
        }

        # scale training and val sequences
        feature_list = list(features.values())
        train_sequences[asset] = [[[((feature_list[idx_inner]['scaling-interval'][1] - feature_list[idx_inner]['scaling-interval'][0]) * (feature - scale_values[asset]['min'][idx_inner]) / (scale_values[asset]['max'][idx_inner] - scale_values[asset]['min'][idx_inner]) + feature_list[idx_inner]['scaling-interval'][0]) 
            for idx_inner, feature in enumerate(features)] for features in interval] for interval in train_sequences[asset]]
        val_sequences[asset] = [[[((feature_list[idx_inner]['scaling-interval'][1] - feature_list[idx_inner]['scaling-interval'][0]) * (feature - scale_values[asset]['min'][idx_inner]) / (scale_values[asset]['max'][idx_inner] - scale_values[asset]['min'][idx_inner]) + feature_list[idx_inner]['scaling-interval'][0]) 
            for idx_inner, feature in enumerate(features)] for features in interval] for interval in val_sequences[asset]]

    # get features for encoder and decoder layer
    layer_features = {
        'encoder_features': [n for n in range(len(features)) if 'enc' in list(features.values())[n]['used-by-layer']],
        'decoder_features': [n for n in range(len(features)) if 'dec' in list(features.values())[n]['used-by-layer']]
    }

    # create Dataloader
    train_ds = CustomDataset(train_sequences, layer_features, params.encoder_input_length, params.prediction_length, shift=params.shift)
    val_ds = CustomDataset(val_sequences, layer_features, params.encoder_input_length, params.prediction_length, params.shift)
    train_dl = DataLoader(train_ds, batch_size=params.batch_size['training'], shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=params.batch_size['validation'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    # Datasets and Dataloader for plots
    full_ds = {}
    full_dl = {}

    for asset_key, asset in data.items():
        full_ds[asset_key] = CustomDataset({asset_key: asset}, layer_features, params.encoder_input_length, params.prediction_length, params.shift)
        full_dl[asset_key] = DataLoader(full_ds[asset_key], batch_size=params.batch_size['plot'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    return train_dl, val_dl, full_dl, scale_values, feature_avg


class InitializeParameters():
    def __init__(self):
        self.val_set_eval_during_training = True
        self.eval_mode = False
        self.load_model = False
        self.overwrite_saved_data = True

        #self.model_name = 'test-multi-asset-5m'
        self.model_name = 'volumetest-allassets2'

        self.asset_interval = 60
        self.split_percent = 0.9
        self.encoder_input_length = 60
        self.prediction_length = 1
        self.shift = 1
        self.lr_overwrite_for_load = None 

        self.batch_size = {
            'training': 64, 
            'validation': 256, 
            'plot': 512
        }
        
        # Hyperparameters
        self.params = {
            # Model
            'encoder_input_length': self.encoder_input_length,
            'n_heads': 2,
            'd_model': 512,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dim_feedforward': 2048, 
            'dropout': 0,
            # Optim
            'optim_name': 'AdamW',
            'optim_lr': 0.0001,
            # Loss
            'loss_name': 'MSELoss'
        }

        self.param_adjust_lr = {
            'lr_decay_factor': 5,
            'loss_decay': 0.01,
            'min_lr': 0.00001
        }
        


def main():
    parameters = InitializeParameters()
    num_intervals = 10000 # num_intervals: number of intervals as integer
    assets = {}

    #asset_codes = ['ZRX', '1INCH', 'AAVE', 'GHST', 'ALGO', 'ANKR', 'ANT', 'REP', 'REPV2', 'AXS', 'BADGER', 'BAL', 'BNT', 'BAND', 'BAT', 'XBT', 'BCH', 'ADA', 'CTSI', 'LINK', 'CHZ', 'COMP', 'ATOM', 'CQT', 'CRV', 'DASH', 'MANA', 'XDG', 'DYDX', 'EWT', 'ENJ', 'MLN', 'EOS', 'ETH', 'ETC', 'FIL', 'FLOW', 'GNO', 'ICX', 'INJ', 'KAR', 'KAVA', 'KEEP', 'KSM', 'KNC', 'LSK', 'LTC', 'LPT', 'LRC', 'MKR', 'MINA', 'MIR', 'XMR', 'MOVR', 'NANO', 'OCEAN', 'OMG', 'OXT', 'OGN', 'OXY', 'PAXG', 'PERP', 'DOT', 'MATIC', 'QTUM', 'REN', 'RARI', 'RAY', 'XRP', 'SRM', 'SDN', 'SC', 'SOL', 'XLM', 'STORJ', 'SUSHI', 'SNX', 'TBTC', 'XTZ', 'GRT', 'SAND', 'TRX', 'UNI', 'WAVES', 'WBTC', 'YFI', 'ZEC']
    asset_codes = ['XBT']
    for asset_code in asset_codes: 
        assets[f'{asset_code}-{parameters.asset_interval}'] = {
            'api-name': f'{asset_code}USD',
            'num_intervals': num_intervals
        }

    features = {
        'open': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': ['enc'] 
        },
        'high': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': ['enc', 'dec'] 

        },
        'low': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': ['enc'] 
        },
        'close': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': ['enc'] 
        },
        'volume': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': ['enc']
        },
        'nbr-of-trades': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': []
        },
        'rsi': {
            'scaling-mode': 'limit-scaler',
            'scaling-interval': [0, 1],
            'scaling-limits': [0, 100],
            'used-by-layer': ['enc']
        },
        'buy-yes': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': ['dec']
        },
        'buy-no': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': ['dec']
        }
    }

    train_dl, val_dl, full_dl, scale_values, feature_avg = data_preprocessing(parameters, assets, features)

    # Start Training and / or Evaluation
    if not parameters.eval_mode:
        # Create model
        if not parameters.load_model:
            trainer = Trainer.create_trainer(params=parameters.params, features=features, scale_values=scale_values)
        else:
            checkpoint = Trainer.load_checkpoint(parameters.model_name)
            trainer = Trainer.create_trainer(params=checkpoint, features=features, scale_values=scale_values)
            trainer.load_training(parameters.model_name)
            if parameters.lr_overwrite_for_load != None: trainer.set_learningrate(parameters.lr_overwrite_for_load)

        for epoch in range(1000):
            print(f' --- Epoch: {epoch + 1}')
            trainer.perform_epoch(dataloader=train_dl, assets=assets, mode='train', feature_avg=feature_avg['train'], param_lr=parameters.param_adjust_lr)
            trainer.save_training(parameters.model_name)

            if parameters.val_set_eval_during_training:
               trainer.perform_epoch(dataloader=val_dl, assets=assets, mode='val', feature_avg=feature_avg['val'])
    else:
        checkpoint = Trainer.load_checkpoint(parameters.model_name)
        trainer = Trainer.create_trainer(params=checkpoint, features=features, scale_values=scale_values)
        trainer.load_training(parameters.model_name)
        trainer.perform_epoch(dataloader=val_dl, assets=assets, mode='val', feature_avg=feature_avg['val'])

        # Plot
        decoder_feature_list = [feature for n, feature in enumerate(features) if 'dec' in list(features.values())[n]['used-by-layer']]
        for asset in full_dl:
            trainer.plot_prediction_vs_target(full_dl[asset], parameters.split_percent, decoder_feature_list)

        breakpoint = None

if __name__ == '__main__':
    main()
