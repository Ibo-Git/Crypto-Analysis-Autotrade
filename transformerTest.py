import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from transformerModel import TransformerModel, Trainer


def main():
    
    feature_size = 2024
    sequences = 50
    batches = 128
    rand = torch.zeros(batches, sequences + 1, feature_size)

    decoder_input = torch.tensor([[list(rand[j][i]) for i in range(len(rand[j]) - 1)] for j in range(batches)])
    target_tensor = torch.tensor([[list(rand[j][i + 1]) for i in range(len(rand[j]) - 1)] for j in range(batches)])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(feature_size, feature_size, n_heads=2, num_decoder_layers=2, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    
    
    for epoch in range(1000):
        loss = trainer.train_transformer(decoder_input, target_tensor)
        print(loss)

if __name__ == '__main__':
    main()