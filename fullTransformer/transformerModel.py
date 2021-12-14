import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
    def __init__(self, feature_size, n_heads, num_encoder_layers, num_decoder_layers, device):
        super(TransformerModel, self).__init__()
        self.device = device

        # Add to model so that save can access them
        self.feature_size = feature_size
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Create model
        self.fc_layer_src = nn.Linear(feature_size, 512)
        self.fc_layer_tgt = nn.Linear(feature_size, 512)
        self.positional_encoder_src = PositionalEncoding(d_model=512, dropout=0.1)
        self.positional_encoder_tgt = PositionalEncoding(d_model=512, dropout=0.1)
        self.transformer = nn.Transformer(d_model=512, nhead=n_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        self.fc_layer_flatten = nn.Linear(512, feature_size)

        self.double()


    def forward(self, src, tgt):
        # transformer masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(nn.Transformer(), tgt.size(dim=1)).to(self.device) # sequence length
        #tgt_padding_mask = torch.ones(tgt.shape)
        #tgt_padding_mask[:, 0:tgt.shape[1] - 1, :] = 0 # sets locations of EOS-token to 1, rest 0

        # forward
        src = self.fc_layer_src(src)
        src = self.positional_encoder_src(src)
        tgt = self.fc_layer_tgt(tgt)
        tgt = self.positional_encoder_tgt(tgt)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        out = self.fc_layer_flatten(out)
        return out

    
class Trainer():
    def __init__(self, model, optimizer, scheduler, criterion, optim_lr, optim_name, loss_name, asset_scaling, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.criterion = criterion
        self.optim_lr = optim_lr
        self.optim_name = optim_name
        self.loss_name = loss_name
        self.asset_scaling = asset_scaling
        self.device = device


    def train_transformer(self, encoder_input, decoder_input, target_tensor):
        self.model.train()

        # backpropagation
        self.optimizer.zero_grad()
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: self.scheduler.step()
        acc = self.get_accuracy(output, target_tensor).to('cpu').detach()
        
        return loss.item(), acc


    def evaluate_transformer(self, encoder_input, decoder_input, target_tensor):
        self.model.eval()

        # determine loss and accuracy
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output, target_tensor)
        acc = self.get_accuracy(output, target_tensor).to('cpu').detach()

        return loss.item(), acc


    def get_accuracy(self, output, target):
        #acc = torch.mean(torch.abs(output - target) [:, 0:target.shape[1] - 1, :] * self.asset_scaling, [0, 1]) # if not used, remove later
        acc = torch.mean(torch.abs(output - target)  * self.asset_scaling, [0, 1])
        return acc


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
            'feature_size': self.model.feature_size,
            'n_heads': self.model.n_heads,
            'num_encoder_layers': self.model.num_encoder_layers,
            'num_decoder_layers': self.model.num_decoder_layers,
            # Optim
            'optim_name': self.optim_name,
            'optim_lr': self.optim_lr,
            # Loss
            'loss_name': self.loss_name,
            # Asset scaling factor
            'asset_scaling': self.asset_scaling
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
        
        feature_size = params['feature_size']
        n_heads = params['n_heads']
        num_encoder_layers = params['num_encoder_layers']
        num_decoder_layers = params['num_decoder_layers']
        optim_lr = params['optim_lr']
        optim_name = params['optim_name']
        loss_name = params['loss_name']
        asset_scaling = params['asset_scaling']

        model = TransformerModel(feature_size=feature_size, n_heads=n_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, device=device).to(device)
        optimizer = None
        criterion = None

        if optim_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=optim_lr)

        if loss_name == 'MSELoss':
            criterion = nn.MSELoss()

        return Trainer(model=model, optimizer=optimizer, scheduler=None, criterion=criterion, optim_lr=optim_lr, optim_name=optim_name, loss_name=loss_name, asset_scaling=asset_scaling, device=device)


    def plot_prediction_vs_target(self, dataloader, split_percent, list_of_features):
        output_for_plot, target_for_plot = self.one_day_prediction_from_dataloader(dataloader)



        #plt.plot(range(len(val_output_for_plot)), val_output_for_plot[:, 0, 0], label = 'Prediction')
        x_axis = range(len(output_for_plot))
        train_index = int(split_percent * x_axis[-1])
        num_features = output_for_plot.shape[-1]
        num_features_x = num_features - num_features // 2
        num_features_y = num_features // 2
        fig, axs = plt.subplots(num_features_x, num_features_y)
        
        n = 0
        for n_x in range(num_features_x):
            for n_y in range(num_features_y):
                axs[n_x, n_y].plot(x_axis[:train_index], target_for_plot[:train_index, 0, n], label = 'Training target')
                axs[n_x, n_y].plot(x_axis[train_index:], target_for_plot[train_index:, 0, n], label = 'Validation target')
                axs[n_x, n_y].plot(x_axis[:train_index], output_for_plot[:train_index, 0, n], label = 'Training prediction')
                axs[n_x, n_y].plot(x_axis[train_index:], output_for_plot[train_index:, 0, n], label = 'Validation prediction')
                axs[n_x, n_y].set_title(f"Prediction vs Target - Feature '{list_of_features[n]}'")
                axs[n_x, n_y].set_xlabel('Time')
                axs[n_x, n_y].set_ylabel('Bitcoin value in USD')
                n += 1
  
        fig.tight_layout()
        plt.show()

    
    # takes entire validation sequence, splits it into multiple sequences and predicts one day for each split 
    def predict_output_from_sequence(self, sequences, encoder_input_length, prediction_length):
        output = []
        target = []

        for n in range(sequences.shape[1] - encoder_input_length - prediction_length):
            encoder_input_for_plot = sequences[:, n:n + encoder_input_length, :]
            target_for_plot = sequences[:, n + encoder_input_length:n + encoder_input_length + prediction_length, :]
            output.append(self.predict_output(encoder_input_for_plot, target_for_plot).to('cpu').detach())
            target.append(target_for_plot)

        output = torch.cat(output)
        target = torch.cat(target)
        output_for_plot = (output[:, 0, 0] * self.asset_scaling).tolist()
        target_for_plot = (target[:, 0, 0] * self.asset_scaling).tolist()

        return output_for_plot, target_for_plot


    def predict_output(self, encoder_input, target_sequence):
        decoder_input = -torch.ones(encoder_input.shape[-1]).unsqueeze(0).unsqueeze(0).double().to(self.device)
        output =  torch.tensor([]).double().to(self.device)

        for n in range(target_sequence.shape[1]):
            output = self.model(encoder_input, torch.cat((decoder_input, output), 1)) # encoder_input [1, 1000, 4]
        
        return output

    
    def one_day_prediction_from_dataloader(self,  dataloader):
        output_for_plot =  torch.tensor([])
        target_for_plot = torch.tensor([])

        for encoder_input, decoder_input, target in dataloader:
            output = self.model(encoder_input, decoder_input).to('cpu').detach()
            output_for_plot = torch.cat((output_for_plot, output))
            target_for_plot = torch.cat((target_for_plot, target))
 
        return output_for_plot * self.asset_scaling, target_for_plot * self.asset_scaling
    
        
        
