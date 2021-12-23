import math
import os
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
from torch.utils import data
from tqdm import tqdm

from NLP_customDataset import TokenIDX


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
    def __init__(self, vocab_size, encoder_input_length, d_model, n_heads, num_encoder_layers, num_decoder_layers, dropout, device):
        super(TransformerModel, self).__init__()
        self.device = device

        # Add to model so that save can access them
        self.vocab_size = vocab_size
        self.encoder_input_length = encoder_input_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

        # 3 different layers use dropout, each layer gets a dropout values such that all dropouts together equal the needed dropout
        dropout = dropout / 3

        # Create model
        self.embedding_src = nn.Embedding(vocab_size, d_model)
        self.embedding_tgt = nn.Embedding(vocab_size, d_model)
        self.positional_encoder_src = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=encoder_input_length)   
        self.positional_encoder_tgt = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=encoder_input_length)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True, dropout=dropout)
        self.fc_layer_flatten = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # transformer masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(nn.Transformer(), tgt.size(dim=1)).to(self.device) # sequence length

        # forward
        src = self.embedding_src(src)
        src = self.positional_encoder_src(src)
        tgt = self.embedding_tgt(tgt)
        tgt = self.positional_encoder_tgt(tgt)
        out = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        out = self.fc_layer_flatten(out)
        return out

    
class Trainer():
    def __init__(self, model, optimizer, scheduler, criterion, optim_lr, optim_name, loss_name, device, bpe_decoder):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.criterion = criterion
        self.optim_lr = optim_lr
        self.optim_name = optim_name
        self.loss_name = loss_name
        self.device = device
        self.bpe_decoder = bpe_decoder


    def perform_epoch(self, dataloader, mode, param_lr=None):
        loss = []
        acc = []
        prev_avg_loss = 0
        curr_avg_loss = 0
        n_batches_lr = len(dataloader) // 4
        pbar = tqdm(dataloader)

        # train or validate an entire epoch
        for num_batch, (encoder_input, decoder_input, expected_output) in enumerate(pbar):
            if mode == 'train':
                batch_loss, batch_acc, output_sequence = self.train_transformer(encoder_input, decoder_input, expected_output)
            elif mode == 'val':
                batch_loss, batch_acc, output_sequence = self.evaluate_transformer(encoder_input, decoder_input, expected_output)                
            
            pbar.set_postfix({ 'loss': batch_loss, 'prev_avg_loss': prev_avg_loss,'avg_loss': curr_avg_loss, 'lr': self.get_learningrate() })
            loss.append(batch_loss)
            acc.append(batch_acc)

            # adjust learning rate during training if parameters are available
            if param_lr is not None:
                if num_batch % n_batches_lr == 0 and num_batch >= 2 * n_batches_lr:
                    prev_avg_loss = np.mean(loss[-2 * n_batches_lr:-n_batches_lr])
                    curr_avg_loss = np.mean(loss[-n_batches_lr:])

                    if (prev_avg_loss - curr_avg_loss) / prev_avg_loss < param_lr['loss_decay']: # 1 = high decay, 0 = no decay
                        curr_lr = self.get_learningrate()
                        self.set_learningrate(max(param_lr['min_lr'], curr_lr / param_lr['lr_decay_factor']))
            

        print(f'{mode}_loss: {np.mean(loss)}\n')
        print(f'{mode}_acc: {np.mean(acc)}\n')       
        print(self.bpe_decoder(output_sequence))
        print(self.bpe_decoder(expected_output[-1].tolist()))

    def train_transformer(self, encoder_input, decoder_input, target_tensor):
        self.model.train()

        # tensors to device
        encoder_input = encoder_input.to(self.device)
        decoder_input = decoder_input.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # backpropagation
        self.optimizer.zero_grad()
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output.reshape(-1, output.size(2)), target_tensor.reshape(-1))
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: self.scheduler.step()
        output_flat = torch.argmax(output, 2)
        acc = self.get_accuracy(output_flat, target_tensor)
        output_sequence = output_flat[-1].tolist()

        return loss.item(), acc, output_sequence


    def evaluate_transformer(self, encoder_input, decoder_input, target_tensor):
        self.model.eval()

        # tensors to device
        encoder_input = encoder_input.to(self.device)
        decoder_input = decoder_input.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # determine loss and accuracy
        output = self.model(encoder_input, decoder_input)
        loss = self.criterion(output.reshape(-1, output.size(2)), target_tensor.reshape(-1)).item()
        output_flat = torch.argmax(output, 2)
        acc = self.get_accuracy(output_flat, target_tensor)
        output_sequence = output_flat[-1].tolist()

        return loss, acc, output_sequence


    def test_transformer(self, encoder_input, generation_length):
        encoder_input = encoder_input.to(self.device)
        output = torch.empty(0, dtype=int).to(self.device)
        decoder_input = torch.tensor([TokenIDX.SOS_IDX]).to(self.device)

        for n in range(generation_length):
            output = self.model(encoder_input.unsqueeze(0), torch.cat((decoder_input, output)).unsqueeze(0))
            output = torch.argmax(output, 2).squeeze(0)

        print(f'{self.bpe_decoder(encoder_input.tolist())[0]}|||{self.bpe_decoder(output.tolist())[0]}')


    def get_accuracy(self, output, target):
        score = []

        for i in range(output.shape[0]):
            score.append(sentence_bleu([output[i].tolist()], target[i].tolist(),  weights=(0.5, 0.5)))

        return np.mean(score)


    def set_learningrate(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr


    def get_learningrate(self):
        for g in self.optimizer.param_groups:
            return g['lr']


    def save_training(self, modelname, processor):            
        if not os.path.isfile(os.path.join(processor.savepath, modelname + '.pt')):
            open(os.path.join(processor.savepath, modelname + '.pt'), 'w')

        torch.save({
            # States
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.get_learningrate(),

            # Hyperparameters
            # Model
            'vocab_size': self.model.vocab_size,
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
        }, os.path.join(processor.savepath, modelname + '.pt'))
    

    def load_checkpoint(modelname, processor):
        if os.path.isfile(os.path.join(processor.savepath, modelname + '.pt')):
            return torch.load(os.path.join(processor.savepath, modelname + '.pt'))
        else:
            return None


    def load_training(self, modelname, processor):
        checkpoint = Trainer.load_checkpoint(modelname, processor=processor)
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.set_learningrate(checkpoint['lr'])
        else:
            print('Model to load does not exist.')


    def create_trainer(params, processor):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        vocab_size = params['vocab_size']
        encoder_input_length = params['encoder_input_length']
        n_heads = params['n_heads']
        d_model = params['d_model']
        num_encoder_layers = params['num_encoder_layers']
        num_decoder_layers = params['num_decoder_layers']
        dropout = params['dropout']
        optim_lr = params['optim_lr']
        optim_name = params['optim_name']
        loss_name = params['loss_name']

        bpe_decoder = processor.bpe.decode

        model = TransformerModel(
            vocab_size=vocab_size, 
            encoder_input_length=encoder_input_length, 
            n_heads=n_heads, 
            d_model=d_model, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dropout=dropout,
            device=device,
        ).to(device)

        optimizer = None
        criterion = None

        if optim_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=optim_lr)

        if loss_name == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()

        return Trainer(model=model, optimizer=optimizer, scheduler=None, criterion=criterion, optim_lr=optim_lr, optim_name=optim_name, loss_name=loss_name, device=device, bpe_decoder=bpe_decoder)

   

    
        
        
