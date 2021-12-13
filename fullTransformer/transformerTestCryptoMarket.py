import math

import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader

from customDataset import CustomDataset
from transformerModel import Trainer, TransformerModel


def data_preprocessing(xbtusd_data, device, encoder_input_length, prediction_length):
    data = []
    for index, row in xbtusd_data.iterrows():
        data.append([
            row['Open'], 
            row['High'], 
            row['Low'], 
            row['Close']
        ])

    train_sequences = data[0:math.floor(len(data) * 0.9)]
    val_sequences = data[math.floor(len(data) * 0.9):]

    train_ds = CustomDataset(train_sequences, encoder_input_length, prediction_length)
    val_ds = CustomDataset(val_sequences, encoder_input_length, prediction_length)
    if device.type == 'cpu':
        train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
        val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)
    else:
        train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        val_dl = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_dl, val_dl, torch.tensor(train_sequences).unsqueeze(0), torch.tensor(val_sequences).unsqueeze(0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_mode = False
    load_model = False
    model_name = 'trained_small3'
    
    # Hyperparameters
    params = {
        # Model
        'feature_size': 4,
        'n_heads': 8,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
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
    
    encoder_input_length = 30
    prediction_length = 1

    train_dl, val_dl, train_sequences, val_sequences = data_preprocessing(xbtusd_data, device, encoder_input_length, prediction_length)
        
    trainer = None

    if not eval_mode:
        # Create model
        if not load_model:
            trainer = Trainer.create_trainer(params=params)
        else:
            checkpoint = Trainer.load_checkpoint(model_name)
            trainer = Trainer.create_trainer(params=checkpoint)
            trainer.load_training(model_name)
        
        for epoch in range(1000):
            print(f' --- Epoch: {epoch + 1}')
            
            trainer.predict_output_from_sequence(val_sequences, encoder_input_length, prediction_length)

            # Train
            loss = []
            acc = []
            for encoder_input, decoder_input, expected_output in train_dl:
                batch_loss, batch_acc = trainer.train_transformer(encoder_input, decoder_input, expected_output)
                loss.append(batch_loss)
                acc.append(batch_acc)

                output = []
                for n in range(train_sequences.shape[1] - encoder_input_length):
                    encoder_input_for_plot = train_sequences[:, n:n + encoder_input_length, :]
                    target_for_plot = train_sequences[:, n + encoder_input_length:n + encoder_input_length + prediction_length, :]
                    output.append(trainer.predict_output(encoder_input_for_plot, target_for_plot))

            print(f'train_loss: {sum(loss) / len(loss)}, train_acc: {(sum(acc) / len(acc)).tolist()}')
            trainer.save_training(model_name)

            # Eval
            loss = []
            acc = []

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

        #n = 100 # starting index for sequence that is fed into transformer. train_sequences has ~2300 sequences. n + encoder_input_length + prediction_length must be < 2300
        #train_encoder_input_for_plot = train_sequences[:, n:n + encoder_input_length, :]
        #train_target_for_plot = train_sequences[:, n + encoder_input_length:n + encoder_input_length + prediction_length, :]
        #trainer.plot_prediction_vs_target(train_encoder_input_for_plot, train_target_for_plot)

        #val_encoder_input_for_plot = val_sequences[:, n:n + encoder_input_length, :]
        #val_target_for_plot = val_sequences[:, n + encoder_input_length:n + encoder_input_length + prediction_length, :]
        #trainer.plot_prediction_vs_target(val_encoder_input_for_plot, val_target_for_plot)

        trainer.plot_prediction_vs_target(val_sequences, encoder_input_length, prediction_length)
        
if __name__ == '__main__':
    main()
