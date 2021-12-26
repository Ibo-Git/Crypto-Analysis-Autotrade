import math
from functools import reduce
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader

from customDataset import CustomDataset
from transformerModel import Trainer, TransformerModel
import pickle
import os
import pandas as pd


def load_asset(assetname, num_intervals, interval):
    # load data
    filename = f'{assetname}_{interval}.csv'
    df= pd.read_csv(
        os.path.join('datasets', 'assets', filename), 
        names=["Datetime", "Open", "High", "Low", "Close", "Volume", "Number of trades"]
    )
    # get number of intervals e.g. for 5m and num_intervals of 17280 you get 60 days of data
    df = df.tail(num_intervals)
    df['Datetime'] = pd.to_datetime(df['Datetime'],unit='s')

    return df


def data_preprocessing(params, assets, features):
    if params.overwrite_saved_data or not os.path.isfile('test.pkl'):
        data = {}

        for asset_key, asset in assets.items():
            #crypto_df = yf.download(tickers=asset['api-name'], period=asset['period'], interval=asset['interval'])
            crypto_df = load_asset(asset['api-name'], num_intervals=asset['num_intervals'], interval=asset['interval'])
            data[asset_key] = []
            last_row_close = 0

            for index, row in crypto_df.iterrows():
                data[asset_key].append([
                    row['Open'], 
                    row['High'], 
                    row['Low'], 
                    row['Close'],
                    1 if last_row_close <= row['Close'] else 0
                ])

                last_row_close = row['Close']

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
            'min': [min(data[asset], key=itemgetter(n))[n] for n in range(features_len)], 
            'max': [max(data[asset], key=itemgetter(n))[n] for n in range(features_len)]
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
        self.model_name = 'test-1_0 copy-1_2l'

        self.split_percent = 0.9
        self.encoder_input_length = 24
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
            'optim_name': 'Adam',
            'optim_lr': 0.0001,
            # Loss
            'loss_name': 'MSELoss'
        }

        self.param_adjust_lr = {
            'lr_decay_factor': 5,
            'loss_decay': 0.05,
            'min_lr': 0.000001
        }
        


def main():
    interval = 5 # possible intervals: 1, 5, 15, 60, 720, 1440
    num_intervals = 2000 # num_intervals: number of intervals as integer
    assets = {
        f'BTC-{interval}': {
            'api-name': 'XBTUSD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'BNB-{interval}': {
            'api-name': 'BNB-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'XRP-{interval}': {
            'api-name': 'XRP-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'SOL1-{interval}': {
            'api-name': 'SOL1-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'ETH-{interval}': {
            'api-name': 'ETH-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'ADA-{interval}': {
            'api-name': 'ADA-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'LUNA1-{interval}': {
            'api-name': 'LUNA1-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'AVAX-{interval}': {
            'api-name': 'AVAX-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'DOGE-{interval}': {
            'api-name': 'DOGE-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'SHIB-{interval}': {
            'api-name': 'SHIB-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'MATIC-{interval}': {
            'api-name': 'MATIC-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'LTC-{interval}': {
            'api-name': 'LTC-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'UNI1-{interval}': {
            'api-name': 'UNI1-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'LINK-{interval}': {
            'api-name': 'LINK-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'ALGO-{interval}': {
            'api-name': 'ALGO-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        },
        f'BCH-{interval}': {
            'api-name': 'BCH-USD',
            'num_intervals': num_intervals,
            'interval': interval, 
        }
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
        'test': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1],
            'used-by-layer': []
        }
    }

    parameters = InitializeParameters()
    train_dl, val_dl, full_dl, scale_values, feature_avg = data_preprocessing(parameters, assets, features)

    # Start Training and / or Evaluation
    trainer = None

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
