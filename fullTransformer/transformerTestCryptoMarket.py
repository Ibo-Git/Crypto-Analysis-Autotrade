import math

import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader

from customDataset import CustomDataset
from transformerModel import Trainer, TransformerModel


def data_preprocessing(xbtusd_data, device):
    train_sequences = xbtusd_data.head(math.floor(len(xbtusd_data) * 0.8))
    val_sequences = xbtusd_data.tail(math.ceil(len(xbtusd_data) * 0.2))
    train_ds = CustomDataset(train_sequences)
    val_ds = CustomDataset(val_sequences)
    if device.type == 'cpu':
        train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
        val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)
    else:
        train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_dl, val_dl



def main():
    eval_mode = False
    load_model = False
    model_name = 'trained'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    params = {
        # Model
        'feature_size': 4,
        'n_heads': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        # Optim
        'optim_name': 'Adam',        
        'optim_lr': 0.0001,
        # Loss
        'loss_name': 'MSELoss',
        'asset_scaling': 100000
    }

    # Data Prep
    xbtusd_data = yf.download(tickers='BTC-USD', period = 'max', interval = '1d')
    xbtusd_data = xbtusd_data / params['asset_scaling']
    train_dl, val_dl = data_preprocessing(xbtusd_data, device)
        
    trainer = None

    if not eval_mode:
        # Create model
        if not load_model:
            trainer = Trainer.create_trainer(params=params)
        else:
            checkpoint = Trainer.load_checkpoint(model_name)
            trainer = Trainer.create_trainer(params=checkpoint)
            trainer.load_training(model_name)
        
        for _ in range(1000):
            # Train
            loss = []
            acc = []
            for encoder_input, decoder_input, expected_output in train_dl:
                batch_loss, batch_acc = trainer.train_transformer(encoder_input, decoder_input, expected_output)
                loss.append(batch_loss)
                acc.append(batch_acc)

            print(f'train_loss: {sum(loss) / len(loss)}, train_acc: {(sum(acc) / len(acc)).tolist()}')
            trainer.save_training(model_name)

            # Eval
            loss = []
            for encoder_input, decoder_input, expected_output in val_dl:
                batch_loss, batch_acc = trainer.evaluate_transformer(encoder_input, decoder_input, expected_output)
                loss.append(batch_loss)
                acc.append(batch_acc)
            
            print(f'val_loss: {sum(loss) / len(loss)}, val_acc: {(sum(acc) / len(acc)).tolist()}')
    else:
        checkpoint = Trainer.load_checkpoint(model_name)
        trainer = Trainer.create_trainer(params=checkpoint)
        trainer.load_training(model_name)

        loss = []
        acc = []

        for encoder_input, decoder_input, expected_output in val_dl:
                batch_loss, batch_acc = trainer.evaluate_transformer(encoder_input, decoder_input, expected_output)
                loss.append(batch_loss)
                acc.append(batch_acc)
            
        print(f'val_loss: {sum(loss) / len(loss)}, val_acc: {(sum(acc) / len(acc)).tolist()}')


if __name__ == '__main__':
    main()