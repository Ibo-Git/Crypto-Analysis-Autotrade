import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader
import math 
from customDataset import CustomDataset
from transformerModel import Trainer, TransformerModel


def data_preprocessing(xbtusd_data):
    train_sequences = xbtusd_data.head(math.floor(len(xbtusd_data) * 0.8))
    val_sequences = xbtusd_data.tail(math.ceil(len(xbtusd_data) * 0.2))
    train_ds = CustomDataset(train_sequences)
    val_ds = CustomDataset(val_sequences)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_dl, val_dl



def main():
    xbtusd_data = yf.download(tickers='BTC-USD', period = 'max', interval = '1d')
    # xbtusd_data = (xbtusd_data - min(xbtusd_data['Low'])) / (100000 - min(xbtusd_data['Low']))   # scales lowest value to 0
    xbtusd_data = xbtusd_data / 100000
    train_dl, val_dl = data_preprocessing(xbtusd_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_feature_size=4, output_feature_size=4, n_heads=2, num_decoder_layers=2, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=None, criterion=criterion, device=device)
    
    
    for epoch in range(1000):
        for decoder_input, expected_output in train_dl:
            loss = trainer.train_transformer(decoder_input, expected_output)
            print(loss)
        for decoder_input, expected_output in val_dl:
            loss = trainer.evaluate_transformer(decoder_input, expected_output)

if __name__ == '__main__':
    main()
