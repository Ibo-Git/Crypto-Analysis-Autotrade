import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from transformerModel import TransformerModel, Trainer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(d_model=6, n_heads=3, num_decoder_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    
    decoder_input = torch.tensor([[[0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]], [[0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]]])
    target_tensor = torch.tensor([[[0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]], [[0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]]])

    for epoch in range(100):
        loss = trainer.train_transformer(decoder_input, target_tensor)
        print(loss)

if __name__ == '__main__':
    main()