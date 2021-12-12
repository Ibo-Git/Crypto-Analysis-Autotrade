import torch
import torch.nn as nn
import math 
import os
import torch.optim as optim

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
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.n_heads = n_heads
        self.num_decoder_layers = num_decoder_layers
        self.device = device

        self.fc_layer_1 = nn.Linear(input_feature_size, 512)
        self.positional_encoder = PositionalEncoding(d_model=512, dropout=0.1)
        decoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)
        self.fc_layer_2 = nn.Linear(512, output_feature_size)
        self.softmax = nn.Softmax(dim=1) # scale output in time between [0, 1]

        self.double()


    def forward(self, tgt):
        # transformer masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(nn.Transformer(), tgt.size(dim=1)).to(self.device) # sequence length
        # forward
        out_fc_1 = self.fc_layer_1(tgt)
        out_pos_enc = self.positional_encoder(out_fc_1)
        # out_dec = self.transformer_decoder(out_pos_enc, tgt_mask)
        out_dec = self.transformer_decoder(out_pos_enc)
        out_fc_2 = self.fc_layer_2(out_dec)
        return out_fc_2

    
class Trainer():
    def __init__(self, model, optimizer, scheduler, criterion, optim_lr, optim_name, loss_name, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.optim_lr = optim_lr
        self.criterion = criterion
        self.optim_name = optim_name
        self.loss_name = loss_name
        self.device = device


    def train_transformer(self, decoder_input, target_tensor):
        self.model.train()
        # tensors to device
        target_tensor = target_tensor.to(self.device).double()
        decoder_input = decoder_input.to(self.device).double()

        # backpropagation
        self.optimizer.zero_grad()
        output = self.model(decoder_input)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: self.scheduler.step()
        return loss.item()


    def evaluate_transformer(self, decoder_input_tensor, target_tensor):
        self.model.eval()

        # tensors to device
        target_tensor = target_tensor.to(self.device).double()
        decoder_input = decoder_input_tensor.to(self.device).double()

        # determine loss and accuracy
        output = self.model(decoder_input)
        loss = self.criterion(output, target_tensor)
        #acc = self.get_accuracy(output, target_tensor)

        return loss.item()


    def get_accuracy(self, output, target):

        return ('Not yet implemented!')


    def set_learningrate(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr


    def save_training(self, modelname):
        if not os.path.isdir('savedFiles'):
            os.makedirs('savedFiles')
            
        if not os.path.isfile(os.path.join('savedFiles', modelname + '.pt')):
            open(os.path.join('savedFiles', modelname + '.pt'), 'w')

        torch.save({
            # States
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

            # Hyperparameters
            # Model
            'input_feature_size': self.model.input_feature_size,
            'output_feature_size': self.model.output_feature_size,
            'n_heads': self.model.n_heads,
            'num_decoder_layers': self.model.num_decoder_layers,
            # Optim
            'optim_name': self.optim_name,
            'optim_lr': self.optim_lr,
            # Loss
            'loss_name': self.loss_name
        }, os.path.join('savedFiles', modelname + '.pt'))
    
    def load_checkpoint(modelname):
        if os.path.isfile(os.path.join('savedFiles', modelname + '.pt')):
            return torch.load(os.path.join('savedFiles', modelname + '.pt'))
        else:
            return None

    def load_training(self, modelname):
        checkpoint = Trainer.load_checkpoint(modelname)
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print('Model to load does not exist.')


    def create_trainer(params):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_feature_size = params['input_feature_size']
        output_feature_size = params['output_feature_size']
        n_heads = params['n_heads']
        num_decoder_layers = params['num_decoder_layers']
        optim_lr = params['optim_lr']
        optim_name = params['optim_name']
        loss_name = params['loss_name']

        model = TransformerModel(input_feature_size=input_feature_size, output_feature_size=output_feature_size, n_heads=n_heads, num_decoder_layers=num_decoder_layers, device=device).to(device)
        optimizer = None
        criterion = None

        if optim_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=optim_lr)

        if loss_name == 'MSELoss':
            criterion = nn.MSELoss()

        return Trainer(model=model, optimizer=optimizer, scheduler=None, criterion=criterion, optim_lr=optim_lr, optim_name=optim_name, loss_name=loss_name, device=device)

