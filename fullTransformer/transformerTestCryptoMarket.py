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


def load_asset(assetname, num_intervals, interval):
    # load data
    filename = f'{assetname}_{interval}.csv'
    df= pd.read_csv(
        os.path.join('datasets', 'assets', filename), 
        names=["Datetime", "Open", "High", "Low", "Close", "Volume", "Number of trades"]
    )

    rsi_window = 10
    df.ta.rsi(close='Close', length=rsi_window, append=True)
    df = df.tail(len(df) - rsi_window)

    # get number of intervals e.g. for 5m and num_intervals of 17280 you get 60 days of data
    df = df.tail(num_intervals)

    return df


def data_preprocessing(params, assets, features):
    if params.overwrite_saved_data or not os.path.isfile('test.pkl'):
        data = {}

        for asset_key, asset in assets.items():
            #crypto_df = yf.download(tickers=asset['api-name'], period=asset['period'], interval=asset['interval'])
            crypto_df = load_asset(asset['api-name'], num_intervals=asset['num_intervals'], interval=asset['interval'])
            data[asset_key] = []

            for index, row in crypto_df.iterrows():
                data[asset_key].append([
                    row['Open'], 
                    row['High'], 
                    row['Low'], 
                    row['Close'],
                    row['Volume'],
                    row['Number of trades'],
                    row['RSI_10']
                ])

        with open('test.pkl', 'wb') as file:
            pickle.dump(data, file)
            
    else:
        with open('test.pkl', 'rb') as file:
            data = pickle.load(file)


    # create copy of data before scaling: used for calculating change in assets
    data_raw = data.copy()
    train_sequences_raw = {}
    val_sequences_raw = {}
    feature_avg_train = {}
    feature_avg_val = {}

    for asset in data:
        train_sequences_raw[asset] = data_raw[asset][0:math.floor(len(data_raw[asset]) * params.split_percent)]
        val_sequences_raw[asset] = data_raw[asset][math.floor(len(data_raw[asset]) * params.split_percent):]
        feature_avg_train[asset] = np.mean(np.abs(np.diff(list(zip(*train_sequences_raw[asset])))), 1)
        feature_avg_val[asset] = np.mean(np.abs(np.diff(list(zip(*val_sequences_raw[asset])))), 1)

    feature_avg = {
        'train': feature_avg_train, 
        'val': feature_avg_val
    }


    # scaling
    scale_values = {}
    train_sequences = {}
    val_sequences = {}
    features_len = len(features.values())

    for asset in data:
        #scale_values[asset] = reduce(lambda l, c: [[min(v[0], c[i]), max(v[1], c[i])] for i, v in enumerate(l)], data[asset], [[9999999999, -9999999999] for _ in range(features_len)])
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

        feature_list = list(features.values())
        data[asset] = [[((feature_list[idx_inner]['scaling-interval'][1] - feature_list[idx_inner]['scaling-interval'][0]) * (feature - scale_values[asset]['min'][idx_inner]) / (scale_values[asset]['max'][idx_inner] - scale_values[asset]['min'][idx_inner]) + feature_list[idx_inner]['scaling-interval'][0]) for idx_inner, feature in enumerate(features)] for features in data[asset]]
        train_sequences[asset] = data[asset][0:math.floor(len(data[asset]) * params.split_percent)]
        val_sequences[asset] = data[asset][math.floor(len(data[asset]) * params.split_percent):]

    layer_features = {
        'encoder_features': [n for n in range(len(features)) if 'enc' in list(features.values())[n]['used-by-layer']],
        'decoder_features': [n for n in range(len(features)) if 'dec' in list(features.values())[n]['used-by-layer']]
    }


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
    interval = 60 # possible intervals: 1, 5, 15, 60, 720, 1440
    num_intervals = 10000 # num_intervals: number of intervals as integer
    assets = {}

    asset_codes = ['ZRX', '1INCH', 'AAVE', 'GHST', 'ALGO', 'ANKR', 'ANT', 'REP', 'REPV2', 'AXS', 'BADGER', 'BAL', 'BNT', 'BAND', 'BAT', 'XBT', 'BCH', 'ADA', 'CTSI', 'LINK', 'CHZ', 'COMP', 'ATOM', 'CQT', 'CRV', 'DASH', 'MANA', 'XDG', 'DYDX', 'EWT', 'ENJ', 'MLN', 'EOS', 'ETH', 'ETC', 'FIL', 'FLOW', 'GNO', 'ICX', 'INJ', 'KAR', 'KAVA', 'KEEP', 'KSM', 'KNC', 'LSK', 'LTC', 'LPT', 'LRC', 'MKR', 'MINA', 'MIR', 'XMR', 'MOVR', 'NANO', 'OCEAN', 'OMG', 'OXT', 'OGN', 'OXY', 'PAXG', 'PERP', 'DOT', 'MATIC', 'QTUM', 'REN', 'RARI', 'RAY', 'XRP', 'SRM', 'SDN', 'SC', 'SOL', 'XLM', 'STORJ', 'SUSHI', 'SNX', 'TBTC', 'XTZ', 'GRT', 'SAND', 'TRX', 'UNI', 'WAVES', 'WBTC', 'YFI', 'ZEC']
    
    for asset_code in asset_codes: 
        assets[f'{asset_code}-{interval}'] = {
            'api-name': f'{asset_code}USD',
            'num_intervals': num_intervals,
            'interval': interval, 
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
        }
    }

    parameters = InitializeParameters()
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
