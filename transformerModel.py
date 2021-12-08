import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, d_model, n_heads, num_decoder_layers):
        super(TransformerModel, self).__init__()
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)
        self.softmax = nn.Softmax(dim=1) # scale output in time between [0, 1]


    def forward(self, tgt):
        # transformer masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(nn.Transformer(), tgt.size(dim=1)) # sequence length
        # forward
        out_dec = self.transformer_decoder(tgt, tgt_mask)
        #out = self.softmax(out_dec)
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

