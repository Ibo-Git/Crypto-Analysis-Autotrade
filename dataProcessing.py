import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import MSELoss
from transformerModel import TransformerModel, Trainer


def main():
    
    feature_size = 2
    rand = torch.rand(4, feature_size)

    decoder_input = torch.tensor([[list(rand[0]), list(rand[1]), list(rand[2])]])
    target_tensor = torch.tensor([[list(rand[1]), list(rand[2]), list(rand[3])]])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(feature_size, feature_size, n_heads=1, num_decoder_layers=2, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.HuberLoss()
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    
    
    for epoch in range(100):
        loss = trainer.train_transformer(decoder_input, target_tensor)
        print(loss)

if __name__ == '__main__':
    main()