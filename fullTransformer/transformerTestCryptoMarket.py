import copy
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

from customDataset import CustomDataset, EvaluationDataset
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


def custom_candles(dataframe, params):
    # create list with each dataframe row in a sublist
    data = dataframe.to_numpy().tolist()

    # concatenate 1 minutes to length of asset_interval and separate overlapping intervals
    data_separated_by_interval = [[data[i + n:i + n + params.asset_interval] for n in range(0, len(data) - params.asset_interval, params.asset_interval)] for i in range(0, params.asset_interval, min(params.copy_shift, params.asset_interval))]

    # combine the 1 minute candles for each feature to one candle of asset_interval length
    for num_copy, shifted_copy in enumerate(data_separated_by_interval):
        for num_candle, candles in enumerate(shifted_copy):
            candles = list(zip(*candles))
            data_separated_by_interval[num_copy][num_candle] = [
                candles[0][-1], # timestamp
                candles[1][0], # open
                max(candles[2]), # high
                min(candles[3]), # low
                candles[4][-1], # close
                sum(candles[5]), # volume
                sum(candles[6]) # number of trades
            ]

    # add additional features e.g. RSI
    rsi_window = 10

    for num_copy, shifted_copy in enumerate(data_separated_by_interval):
        # define temporary dataframe to determine additional features using pandas_ta
        df_temp = pd.DataFrame(shifted_copy, columns=["Datetime", "Open", "High", "Low", "Close", "Volume", "Number of trades"])
        df_temp.ta.rsi(close='Close', length=rsi_window, append=True)
        df_temp = df_temp.tail(len(df_temp) - rsi_window) # remove NaN entries 
        data_temp = {} # temporary list to convert dataframe to list
        data_temp['Data'] = []
        data_temp['Datetime'] = []

        last_row = None

        for _, row in df_temp.iterrows():
            if last_row is not None:
                # add more features if needed
                data_temp['Data'].append([
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

                data_temp['Datetime'].append(int(row['Datetime']))

            last_row = row
        
        data_separated_by_interval[num_copy] = data_temp

    return data_separated_by_interval


def data_preprocessing(params, assets, features):
    data = {}
    data_df_1min = {}
    assets_cleaned = []

    # load each asset using custom candles
    if params.overwrite_saved_data or not os.path.isfile('data.pkl'):
        for asset_key, asset in assets.items():
            data_1min = load_asset(asset['api-name'], num_intervals=asset['num_intervals'])
            if len(data_1min) == params.num_intervals:
                data_df_1min[asset_key] = data_1min.tail(params.split_percent*len(data_1min))
                data[asset_key] = custom_candles(data_df_1min[asset_key], params)
                assets_cleaned.append(asset_key)

        with open('data.pkl', 'wb') as file:
            pickle.dump([data, data_df_1min, assets_cleaned], file)
            
    else:
        with open('data.pkl', 'rb') as file:
            data, data_df_1min, assets_cleaned = pickle.load(file)

    # split into training and validation sequences
    train_sequences = {}
    val_sequences = {}
    feature_avg = {
        'train': {},
        'val': {}
    }

    for asset_key, asset in data.items():
        train_sequences[asset_key] = {}
        val_sequences[asset_key] = {}
        train_sequences[asset_key]['Data']  = []
        train_sequences[asset_key]['Datetime']  = []
        val_sequences[asset_key]['Data']  = []
        val_sequences[asset_key]['Datetime']  = []
        shifted_copy_avg_train = []
        shifted_copy_avg_val = []

        for num_copy, shifted_copy in enumerate(asset):
            for key, value in shifted_copy.items():
                train_sequences[asset_key][key].append(value[0:math.floor(len(value) * params.split_percent)])
                val_sequences[asset_key][key].append(value[math.floor(len(value) * params.split_percent):])

            # determine average change for each shifted copy
            shifted_copy_avg_train.append(np.mean(np.abs(np.diff(list(zip(*train_sequences[asset_key]['Data'][num_copy])))), 1))
            shifted_copy_avg_val.append(np.mean(np.abs(np.diff(list(zip(*val_sequences[asset_key]['Data'][num_copy])))), 1))

        # average the avg change over all shifted copies
        feature_avg['train'] [asset_key] = np.mean(list(zip(*shifted_copy_avg_train)), 1)
        feature_avg['val'][asset_key] = np.mean(list(zip(*shifted_copy_avg_val)), 1)
    
    # scaling training and validation sequences according to their scaling-mode
    features_len = len(features.values())
    feature_list = list(features.values())
    separated_features = {}
    scale_values = {}

    for asset in data:
        # separate all features using zip and concatenate all tuples using reduce
        separated_features[asset] = [reduce(lambda x,y: x+y, list(zip(*[list(zip(*shifted_copy['Data'])) for shifted_copy in data[asset]]))[n]) for n in range(features_len)]
        min_values = []
        max_values = []
        
        # determine min and maxm values according to the scaling-mode
        for n, feature in enumerate(separated_features[asset]):
            if feature_list[n]['scaling-mode'] == 'min-max-scaler':
                min_value = min(feature)
                max_value = max(feature)

            elif feature_list[n]['scaling-mode'] == 'limit-scaler':
                min_value = feature_list[n]['scaling-limits'][0]
                max_value = feature_list[n]['scaling-limits'][1]

            min_values.append(min_value),
            max_values.append(max_value)

        scale_values[asset] = {'min': min_values, 'max': max_values}

        # scale training and val sequences using the following formula: (b-a)*(x-min)/(max-min)+a where [a, b] are the scaling limits e.g. [0, 1] and min/max the extreme values
        train_sequences[asset]['Data'] = [[[
            (feature_list[idx_inner]['scaling-interval'][1] - feature_list[idx_inner]['scaling-interval'][0]) * (feature - scale_values[asset]['min'][idx_inner]) / (scale_values[asset]['max'][idx_inner] - scale_values[asset]['min'][idx_inner]) + feature_list[idx_inner]['scaling-interval'][0]
        for idx_inner, feature in enumerate(features)] for features in shifted_copy] for shifted_copy in train_sequences[asset]['Data']]
       
        val_sequences[asset]['Data'] = [[[
            (feature_list[idx_inner]['scaling-interval'][1] - feature_list[idx_inner]['scaling-interval'][0]) * (feature - scale_values[asset]['min'][idx_inner]) / (scale_values[asset]['max'][idx_inner] - scale_values[asset]['min'][idx_inner]) + feature_list[idx_inner]['scaling-interval'][0]
        for idx_inner, feature in enumerate(features)] for features in shifted_copy] for shifted_copy in val_sequences[asset]['Data']]

    # get features for encoder and decoder layer
    layer_features = {
        'encoder_features': [n for n in range(features_len) if 'enc' in feature_list[n]['used-by-layer']],
        'decoder_features': [n for n in range(features_len) if 'dec' in feature_list[n]['used-by-layer']]
    }

    # create Dataloader
    train_ds = CustomDataset(train_sequences, layer_features, params.encoder_input_length, params.prediction_length, shift=params.sequence_shift)
    eval_ds = EvaluationDataset(val_sequences, layer_features, params.encoder_input_length, params.prediction_length, shift=params.sequence_shift)
    val_ds = CustomDataset(val_sequences, layer_features, params.encoder_input_length, params.prediction_length, params.sequence_shift)
    train_dl = DataLoader(train_ds, batch_size=params.batch_size['training'], shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=params.batch_size['validation'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)
    eval_dl = DataLoader(eval_ds, batch_size=params.batch_size['validation'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    # Datasets and Dataloader for plots
    full_ds = {}
    full_dl = {}

    # for asset_key, asset in data.items():
    #     full_ds[asset_key] = CustomDataset({asset_key: asset}, layer_features, params.encoder_input_length, params.prediction_length, params.sequence_shift)
    #     full_dl[asset_key] = DataLoader(full_ds[asset_key], batch_size=params.batch_size['plot'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    return train_dl, val_dl, eval_dl, scale_values, feature_avg, assets_cleaned, data_df_1min


class InitializeParameters():
    def __init__(self):
        self.val_set_eval_during_training = True
        self.eval_mode = True
        self.load_model = True
        self.overwrite_saved_data = False

        #self.model_name = 'test-multi-asset-5m'
        self.model_name = 'test--1'

        self.asset_interval = 60
        self.copy_shift = 4
        self.num_intervals = 2000 * self.asset_interval # num_intervals: number of intervals as integer
        self.split_percent = 0.9
        self.encoder_input_length = 4
        self.prediction_length = 1
        self.sequence_shift = 4
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
            'n_heads': 8,
            'd_model': 512,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
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
    assets = {}

    asset_codes = ['ZRX', '1INCH', 'AAVE', 'GHST', 'ALGO', 'ANKR', 'ANT', 'REP', 'REPV2', 'AXS', 'BADGER', 'BAL', 'BNT', 'BAND', 'BAT', 'XBT', 'BCH', 'ADA', 'CTSI', 'LINK', 'CHZ', 'COMP', 'ATOM', 'CQT', 'CRV', 'DASH', 'MANA', 'XDG', 'DYDX', 'EWT', 'ENJ', 'MLN', 'EOS', 'ETH', 'ETC', 'FIL', 'FLOW', 'GNO', 'ICX', 'INJ', 'KAR', 'KAVA', 'KEEP', 'KSM', 'KNC', 'LSK', 'LTC', 'LPT', 'LRC', 'MKR', 'MINA', 'MIR', 'XMR', 'MOVR', 'NANO', 'OCEAN', 'OMG', 'OXT', 'OGN', 'OXY', 'PAXG', 'PERP', 'DOT', 'MATIC', 'QTUM', 'REN', 'RARI', 'RAY', 'XRP', 'SRM', 'SDN', 'SC', 'SOL', 'XLM', 'STORJ', 'SUSHI', 'SNX', 'TBTC', 'XTZ', 'GRT', 'SAND', 'TRX', 'UNI', 'WAVES', 'WBTC', 'YFI', 'ZEC']
    #asset_codes = ['XBT', 'ETH', 'RAY']
    for asset_code in asset_codes: 
        assets[f'{asset_code}-{parameters.asset_interval}'] = {
            'api-name': f'{asset_code}USD',
            'num_intervals': parameters.num_intervals
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
            'used-by-layer': ['enc'] 

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

    train_dl, val_dl, eval_dl, scale_values, feature_avg, assets_cleaned, data_df_1min = data_preprocessing(parameters, assets, features)

    # remove assets from dict that didn't contain much data
    for asset_key in asset_codes:
        if f'{asset_key}-{parameters.asset_interval}' not in assets_cleaned:
            del assets[f'{asset_key}-{parameters.asset_interval}']

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
        # # # # # # # # trainer.perform_epoch(dataloader=val_dl, assets=assets, mode='val', feature_avg=feature_avg['val'])

        prediction_map = trainer.map_prediction_to_1min(eval_dl, assets)
        buy_in_points = { asset_key: [] for asset_key in assets.keys() }
        percentage_earnings = { asset_key: 0 for asset_key in assets.keys() }
        percentage_earnings_list = { asset_key: [] for asset_key in assets.keys() }
        
        last_datapoint = { asset_key: None for asset_key in assets.keys() }

        for asset_key, data_1min in data_df_1min.items():
            for _, datapoint_1min in data_1min.iterrows():
                prediction_map_key = int(datapoint_1min['Datetime'])

                for buy_in_point in buy_in_points[asset_key]:
                    if buy_in_point['state'] == 'open':
                        if last_datapoint[asset_key] is not None and (buy_in_point['entry_datapoint']['Close'] > datapoint_1min['Close'] * 1.004 or last_datapoint[asset_key]['Close'] > datapoint_1min['Close'] * 1.003):
                            buy_in_point['state'] = 'closed'
                            percent_earning = (datapoint_1min['Close'] * 0.9974**2 - buy_in_point['entry_datapoint']['Close']) / buy_in_point['entry_datapoint']['Close']
                            percentage_earnings_list[asset_key].append(percent_earning)
                            percentage_earnings[asset_key] += percent_earning
                            buy_in_point['last_datapoint'] = datapoint_1min
                        # if last_datapoint[asset_key] is not None and last_datapoint[asset_key]['Close'] > datapoint_1min['Close']:
                        #     buy_in_point['state'] = 'closed'
                        #     percent_earning = (datapoint_1min['Close'] * 0.9974**2 - buy_in_point['entry_datapoint']['Close']) / buy_in_point['entry_datapoint']['Close']
                        #     percentage_earnings_list[asset_key].append(percent_earning)
                        #     percentage_earnings[asset_key] += percent_earning
                        #     buy_in_point['last_datapoint'] = datapoint_1min
                        # if last_datapoint[asset_key] is not None and last_datapoint[asset_key]['Close'] > (datapoint_1min['High'] - datapoint_1min['Close']) * 0.5 + datapoint_1min['Close']:
                        #     buy_in_point['state'] = 'closed'
                        #     percent_earning = (datapoint_1min['High'] * 0.9974**2 - buy_in_point['entry_datapoint']['Close']) / buy_in_point['entry_datapoint']['Close']
                        #     percentage_earnings_list[asset_key].append(percent_earning)
                        #     percentage_earnings[asset_key] += percent_earning
                        #     buy_in_point['last_datapoint'] = datapoint_1min

                if prediction_map_key in prediction_map[asset_key]:
                    prediction = prediction_map[asset_key][prediction_map_key]
                    buy_yes = prediction[0] >= 0.78 and prediction[1] <= 0.22

                    if buy_yes:
                        buy_in_points[asset_key].append({ 'timestamp': prediction_map_key, 'entry_datapoint': datapoint_1min, 'state': 'open' })

                last_datapoint[asset_key] = datapoint_1min


        # # Plot
        # decoder_feature_list = [feature for n, feature in enumerate(features) if 'dec' in list(features.values())[n]['used-by-layer']]
        # for asset in full_dl:
        #     trainer.plot_prediction_vs_target(full_dl[asset], parameters.split_percent, decoder_feature_list)

        breakpoint = None

if __name__ == '__main__':
    main()
