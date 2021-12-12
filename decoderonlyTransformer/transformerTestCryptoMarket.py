import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader
import math 
from customDataset import CustomDataset
from transformerModel import Trainer, TransformerModel


def data_preprocessing(xbtusd_data):
    train_sequences = xbtusd_data.head(math.floor(len(xbtusd_data) * 0.98))
    val_sequences = xbtusd_data.tail(math.ceil(len(xbtusd_data) * 0.02))
    train_ds = CustomDataset(train_sequences)
    val_ds = CustomDataset(val_sequences)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_dl, val_dl



def main():
    eval_mode = False
    load_model = False
    model_name = 'split98-2'
    #model_name = 'trained copy_0-0004_0-0037'

    # Hyperparameters
    params = {
        # Model
        'input_feature_size': 4,
        'output_feature_size': 4,
        'n_heads': 2,
        'num_decoder_layers': 1,
        # Optim
        'optim_name': 'Adam',
        'optim_lr': 0.0001,
        # Loss
        'loss_name': 'MSELoss'
    }

    # Data Prep
    xbtusd_data = yf.download(tickers='BTC-USD', period = 'max', interval = '1d')
    xbtusd_data = xbtusd_data / 100000
    train_dl, val_dl = data_preprocessing(xbtusd_data)

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

            for decoder_input, expected_output in train_dl:
                batch_loss, batch_acc = trainer.train_transformer(decoder_input, expected_output)
                loss.append(batch_loss)
                acc.append(batch_acc)

            print(f'train_loss: {sum(loss) / len(loss)}, train_acc: {(sum(acc) / len(acc)).tolist()}')
            trainer.save_training(model_name)

            # Eval
            loss = []
            acc = []

            for decoder_input, expected_output in val_dl:
                batch_loss, batch_acc = trainer.evaluate_transformer(decoder_input, expected_output)
                loss.append(batch_loss)
                acc.append(batch_acc)
            
            print(f'val_loss: {sum(loss) / len(loss)}, val_acc: {(sum(acc) / len(acc)).tolist()}')
    else:
        checkpoint = Trainer.load_checkpoint(model_name)
        trainer = Trainer.create_trainer(params=checkpoint)
        trainer.load_training(model_name)

        loss = []
        acc = []

        for decoder_input, expected_output in val_dl:
                batch_loss, batch_acc = trainer.evaluate_transformer(decoder_input, expected_output)
                loss.append(batch_loss)
                acc.append(batch_acc.detach())
            
        print(f'val_loss: {sum(loss) / len(loss)}, val_acc: {(sum(acc) / len(acc)).tolist()}')

if __name__ == '__main__':
    main()
