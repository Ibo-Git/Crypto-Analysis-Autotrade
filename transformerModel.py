import torch
import torch.nn as nn
import math 


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_feature_size, output_feature_size, n_heads, num_decoder_layers, device):
        super(TransformerModel, self).__init__()
        self.device = device
        self.positional_encoder = PositionalEncoding(input_feature_size, dropout=0.1)
        decoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)
        self.fc_layer = nn.Linear(512, output_feature_size)
        self.softmax = nn.Softmax(dim=1) # scale output in time between [0, 1]


    def forward(self, tgt):
        # transformer masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(nn.Transformer(), tgt.size(dim=1)).to(self.device) # sequence length
        # forward
        out_dec = self.transformer_decoder(tgt, tgt_mask)
        out = self.softmax(out_dec)
        print(out)
        return out_dec

    
class Trainer():
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device


    def train_transformer(self, decoder_input, target_tensor):
        # set mode
        self.model.train()

        # tensors to device
        target_tensor = target_tensor.to(self.device)
        decoder_input = decoder_input.to(self.device)

        # backpropagation
        self.optimizer.zero_grad()
        output = self.model(decoder_input)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def evaluate_transformer(self, decoder_input_tensor, target_tensor):
        # set mode
        self.model.eval()

        # tensors to device
        target_tensor = target_tensor.to(self.device)
        decoder_input = decoder_input_tensor.to(self.device)

        # determine loss and accuracy
        output = self.model(decoder_input)
        loss = self.criterion(output, target_tensor)
        acc = self.get_accuracy()

        return loss.item(), acc


    def get_accuracy(self, output, target):

        return

