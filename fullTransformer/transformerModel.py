import math
import os
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm


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
        x = x + self.pe[:x.size(1), :].permute(1, 0, 2)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, feature_size, encoder_input_length, d_model, n_heads, num_encoder_layers, num_decoder_layers, dropout, device):
        super(TransformerModel, self).__init__()
        self.device = device

        # Add to model so that save can access them
        self.feature_size = feature_size
        self.encoder_input_length = encoder_input_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

        # 3 different layers use dropout, each layer gets a dropout values such that all dropouts together equal the needed dropout
        dropout = dropout / 3

        # Create model
        self.fc_layer_src = nn.Linear(feature_size, d_model)
        self.fc_layer_tgt = nn.Linear(feature_size, d_model)
        self.positional_encoder_src = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=encoder_input_length)   
        self.positional_encoder_tgt = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=encoder_input_length)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True, dropout=dropout)
        self.fc_layer_flatten = nn.Linear(d_model, feature_size)

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
    def __init__(self, model, optimizer, scheduler, criterion, optim_lr, optim_name, loss_name, features, scale_values, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.criterion = criterion
        self.optim_lr = optim_lr
        self.optim_name = optim_name
        self.loss_name = loss_name
        self.device = device
        self.features = features
        self.scale_values = scale_values


    def perform_epoch(self, dataloader, assets, mode, param_lr=None):
        loss = []
        acc = {}
        curr_avg_loss = 0
        pbar = tqdm(dataloader)

        for asset in assets: acc[asset] = []
        
        # train or validate an entire epoch
        for num_batch, (encoder_input, decoder_input, expected_output, asset_tag) in enumerate(pbar):
            if mode == 'train':
                batch_loss, batch_acc = self.train_transformer(encoder_input, decoder_input, expected_output, asset_tag)
            elif mode == 'val':
                batch_loss, batch_acc = self.evaluate_transformer(encoder_input, decoder_input, expected_output, asset_tag)                
            
            pbar.set_postfix({'loss': batch_loss, 'avg_loss': curr_avg_loss, 'curr_lr': self.get_learningrate()})
            loss.append(batch_loss)

            # adjust learning rate during training if parameters are available
            if param_lr is not None:
                if num_batch % param_lr['n_batches_lr'] == 0 and num_batch >= 2 * param_lr['n_batches_lr']:
                    prev_avg_loss = np.mean(loss[-2 * param_lr['n_batches_lr']:-param_lr['n_batches_lr']])
                    curr_avg_loss = np.mean(loss[-param_lr['n_batches_lr']:])

                    if (prev_avg_loss - curr_avg_loss) / prev_avg_loss < param_lr['loss_decay']: # 1 = high decay, 0 = no decay
                        curr_lr = self.get_learningrate()
                        self.set_learningrate(curr_lr / param_lr['lr_decay_factor'])
                
                    pbar.set_postfix({'avg_loss': curr_avg_loss})

            
            # get accuracy for every asset separately
            for asset in list(set(asset_tag)):
                acc[asset].append(batch_acc[asset])

        print(f'{mode}_loss: {np.mean(loss)}\n')

        for asset in assets:
            print(f'{mode}_acc_{asset}: {np.mean(acc[asset], 0)}\n')
            

    def train_transformer(self, encoder_input, decoder_input, target_tensor, asset_tag):
        self.model.train()

        # tensors to device
        encoder_input = encoder_input.to(self.device)
        decoder_input = decoder_input.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # backpropagation
        self.optimizer.zero_grad()
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: self.scheduler.step()
        acc = self.get_accuracy(output, target_tensor, asset_tag)
        
        return loss.item(), acc


    def evaluate_transformer(self, encoder_input, decoder_input, target_tensor, asset_tag):
        self.model.eval()

        # tensors to device
        encoder_input = encoder_input.to(self.device)
        decoder_input = decoder_input.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # determine loss and accuracy
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output, target_tensor).item()
        acc = self.get_accuracy(output, target_tensor, asset_tag)

        return loss, acc


    def get_accuracy(self, output, target, asset_tag):
        scaled_output = self.scale_assets_to_normal(output, asset_tag)
        scaled_target = self.scale_assets_to_normal(target, asset_tag)
        assets = set(asset_tag)
        acc = {}

        for asset in assets:
            entries_matching = np.array(asset_tag) == np.array(list([asset]) * len(asset_tag))
            acc[asset] = torch.mean(torch.abs(scaled_output[entries_matching, :, :] - scaled_target[entries_matching, :, :]), [0, 1]).tolist()

        return acc

    def scale_assets_to_normal(self, data, asset_tag):
        scale_values = self.scale_values
        features = self.features
        feature_list = list(features.values())

        # a = feature_list[idx_inner]['scaling-interval'][0]
        # b = feature_list[idx_inner]['scaling-interval'][1]
        # min = scale_values[asset_tag[idx_sequence]]['min'][idx_inner]
        # max = scale_values[asset_tag[idx_sequence]]['max'][idx_inner]
        # scaled_data = (data - a) / (b - a) * (max - min) + min        with interval [a, b] 

        scaled_data = [[[((feature - feature_list[idx_inner]['scaling-interval'][0]) / (feature_list[idx_inner]['scaling-interval'][1] - feature_list[idx_inner]['scaling-interval'][0]) * (scale_values[asset_tag[idx_sequence]]['max'][idx_inner] - scale_values[asset_tag[idx_sequence]]['min'][idx_inner]) + scale_values[asset_tag[idx_sequence]]['min'][idx_inner]) for idx_inner, feature in enumerate(features)] for features in sequence] for idx_sequence, sequence in enumerate(data)]

        return torch.tensor(scaled_data)


    def set_learningrate(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr

    def get_learningrate(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def save_training(self, modelname):
        if not os.path.isdir('savedFiles'):
            os.makedirs('savedFiles')
            
        if not os.path.isfile(os.path.join('savedFiles', modelname + '.pt')):
            open(os.path.join('savedFiles', modelname + '.pt'), 'w')

        torch.save({
            # States
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.get_learningrate(),

            # Hyperparameters
            # Model
            'feature_size': self.model.feature_size,
            'encoder_input_length': self.model.encoder_input_length,
            'n_heads': self.model.n_heads,
            'd_model': self.model.d_model,
            'num_encoder_layers': self.model.num_encoder_layers,
            'num_decoder_layers': self.model.num_decoder_layers,
            'dropout': self.model.dropout,
            # Optim
            'optim_name': self.optim_name,
            'optim_lr': self.optim_lr,
            # Loss
            'loss_name': self.loss_name,
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
            self.set_learningrate(checkpoint['lr'])
        else:
            print('Model to load does not exist.')


    def create_trainer(params, features, scale_values):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        feature_size = params['feature_size']
        encoder_input_length = params['encoder_input_length']
        n_heads = params['n_heads']
        d_model = params['d_model']
        num_encoder_layers = params['num_encoder_layers']
        num_decoder_layers = params['num_decoder_layers']
        dropout = params['dropout']
        optim_lr = params['optim_lr']
        optim_name = params['optim_name']
        loss_name = params['loss_name']      

        model = TransformerModel(
            feature_size=feature_size, 
            encoder_input_length=encoder_input_length, 
            n_heads=n_heads, 
            d_model=d_model, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dropout=dropout,
            device=device
        ).to(device)

        optimizer = None
        criterion = None

        if optim_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=optim_lr)

        if loss_name == 'MSELoss':
            criterion = nn.MSELoss()

        return Trainer(model=model, optimizer=optimizer, scheduler=None, criterion=criterion, optim_lr=optim_lr, optim_name=optim_name, loss_name=loss_name, features=features, scale_values=scale_values, device=device)


    def plot_prediction_vs_target(self, dataloader, split_percent, list_of_features):

        # get x and y data for plots
        output_for_plot, target_for_plot, asset_tag = self.one_day_prediction_from_dataloader(dataloader)
        x_axis = range(len(output_for_plot))
        train_index = int(split_percent * x_axis[-1])
        
        # create subplots for all features
        num_features = output_for_plot.shape[-1]
        num_features_x = num_features - num_features // 2
        num_features_y = max(1, num_features // 2)
        fig, axs = plt.subplots(num_features_x, num_features_y)
        
        n = 0
        for n_x in range(num_features_x):
            for n_y in range(num_features_y):
                axs[n_x, n_y].plot(x_axis[:train_index], target_for_plot[:train_index, 0, n], label = 'Training target')
                axs[n_x, n_y].plot(x_axis[train_index:], target_for_plot[train_index:, 0, n], label = 'Validation target')
                axs[n_x, n_y].plot(x_axis[:train_index], output_for_plot[:train_index, 0, n], label = 'Training prediction')
                axs[n_x, n_y].plot(x_axis[train_index:], output_for_plot[train_index:, 0, n], label = 'Validation prediction')
                axs[n_x, n_y].legend(loc = "upper left")
                axs[n_x, n_y].set_title(f"Prediction vs Target - Feature '{list_of_features[n]}'")
                axs[n_x, n_y].set_xlabel('Time')
                axs[n_x, n_y].set_ylabel('Bitcoin value in USD')
                n += 1
  
        fig.tight_layout()
        fig.suptitle(asset_tag)
        plt.show()
    
    
    def one_day_prediction_from_dataloader(self,  dataloader):
        output_for_plot =  torch.tensor([])
        target_for_plot = torch.tensor([])

        for encoder_input, decoder_input, target, asset_tag in dataloader:
            output = self.model(encoder_input.to(self.device), decoder_input.to(self.device))

            #  scale values back to normal
            #output = self.scale_assets_to_normal(output, asset_tag)
            #target = self.scale_assets_to_normal(target, asset_tag)

            # concatenate output and target to one single tensor
            output_for_plot = torch.cat((output_for_plot, output.to('cpu').detach()))
            target_for_plot = torch.cat((target_for_plot, target.to('cpu').detach()))
        
        return output_for_plot, target_for_plot, asset_tag[0]

    
   

    
        
        
