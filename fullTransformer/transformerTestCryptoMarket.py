import math
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader

from customDataset import CustomDataset
from transformerModel import Trainer, TransformerModel


def data_preprocessing(params, assets, features):
    data = {}
    
    for asset_key, asset in assets.items():
        crypto_df = yf.download(tickers=asset['api-name'], period=asset['period'], interval=asset['interval'])
        data[asset_key] = []

        for index, row in crypto_df.iterrows():
            data[asset_key] .append([
                row['Open'], 
                row['High'], 
                row['Low'], 
                row['Close']
            ])
    
    # scaling
    scale_values = {}
    train_sequences = {}
    val_sequences = {}

    for asset in data:
        scale_values[asset] = {
            'min': reduce(min, data[asset]), 
            'max': reduce(max, data[asset])
        }

        feature_list = list(features.values())
        data[asset] = [[((feature_list[idx_inner]['scaling-interval'][1] - feature_list[idx_inner]['scaling-interval'][0]) * (feature - scale_values[asset]['min'][idx_inner]) / (scale_values[asset]['max'][idx_inner] - scale_values[asset]['min'][idx_inner]) + feature_list[idx_inner]['scaling-interval'][0]) for idx_inner, feature in enumerate(features)] for features in data[asset]]
        train_sequences[asset] = data[asset][0:math.floor(len(data[asset]) * params.split_percent)]
        val_sequences[asset] = data[asset][math.floor(len(data[asset]) * params.split_percent):]

    train_ds = CustomDataset(train_sequences, params.encoder_input_length, params.prediction_length)
    val_ds = CustomDataset(val_sequences, params.encoder_input_length, params.prediction_length)
    train_dl = DataLoader(train_ds, batch_size=params.batch_size['training'], shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=params.batch_size['validation'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    # Datasets and Dataloader for plots
    full_ds = {}
    full_dl = {}

    for asset_key, asset in data.items():
        full_ds[asset_key] = CustomDataset({asset_key: asset}, params.encoder_input_length, params.prediction_length)
        full_dl[asset_key] = DataLoader(full_ds[asset_key], batch_size=params.batch_size['plot'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    return train_dl, val_dl, full_dl, scale_values


class InitializeParameters():
    def __init__(self):
        self. val_set_eval_during_training = False
        self.eval_mode = False
        self.load_model = False
        #self.model_name = 'test-multi-asset-5m'
        self.model_name = 'test-multi-asset-2m'

        self.split_percent = 0.9
        self.encoder_input_length = 12
        self.prediction_length = 1
        self.lr_overwrite_for_load = 0.0001 

        self.batch_size = {
            'training': 256, 
            'validation': 256, 
            'plot': 512
        }
        
        # Hyperparameters
        self.params = {
            # Model
            'feature_size': 4,
            'encoder_input_length': self.encoder_input_length,
            'n_heads': 2,
            'd_model': 512,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dropout': 0,
            # Optim
            'optim_name': 'Adam',
            'optim_lr': 0.0001,
            # Loss
            'loss_name': 'MSELoss'
        }

        self.param_adjust_lr = {
            'n_batches_lr': 20,
            'lr_decay_factor': 10,
            'loss_decay': 0.1,
        }
        


def main():
    # possible intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    # period depends on interval: 'max' for intervals > 1d, '60d' for 1d < interval < 1m, and for 1m set to 7d 
    assets = {
        'BTC-2m': {
            'api-name': 'BTC-USD',
            'period': '60d',
            'interval': '2m', 
        },
        'ETH-2m': {
            'api-name': 'ETH-USD',
            'period': '60d',
            'interval': '2m', 
        },
        'BNB-2m': {
            'api-name': 'BNB-USD',
            'period': '60d',
            'interval': '2m', 
        },
        'SOL1-2m': {
            'api-name': 'SOL1-USD',
            'period': '60d',
            'interval': '2m', 
        },
        'XRP-2m': {
            'api-name': 'XRP-USD',
            'period': '60d',
            'interval': '2m', 
        }
    }

    features = {
        'open': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1]
        },
        'high': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1]
        },
        'low': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1]
        },
        'close': {
            'scaling-mode': 'min-max-scaler',
            'scaling-interval': [0, 1]
        }
    }

    parameters = InitializeParameters()
    train_dl, val_dl, full_dl, scale_values = data_preprocessing(parameters, assets, features)
    
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
            trainer.perform_epoch(train_dl, assets, 'train', parameters.param_adjust_lr)
            trainer.save_training(parameters.model_name)

            if parameters.val_set_eval_during_training:
               trainer.perform_epoch(val_dl, assets, 'val')
    else:
        checkpoint = Trainer.load_checkpoint(parameters.model_name)
        trainer = Trainer.create_trainer(params=checkpoint, features=features, scale_values=scale_values)
        trainer.load_training(parameters.model_name)
        trainer.perform_epoch(val_dl, assets, 'val')

        # Plot
        for asset in full_dl:
            trainer.plot_prediction_vs_target(full_dl[asset], parameters.split_percent, list(features.keys()))

        breakpoint = None

if __name__ == '__main__':
    main()
