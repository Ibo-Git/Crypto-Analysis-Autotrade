from json import decoder

import torch
from torch.utils.data import Dataset
import random
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, layer_features, encoder_input_length=50, prediction_length=1, shift=1, dataset=None):
        
        # split entire history into sequences
        sequence_length = encoder_input_length + prediction_length
        asset_tag = []
        sequences = []

        for asset_key, asset in data.items():
            for asset_overlap in asset['Data']:
                temp_sequence_length = len(sequences)
                sequences = sequences + [asset_overlap[n:n + sequence_length] for n in range(0, len(asset_overlap) - sequence_length, min(shift, sequence_length))]
                asset_tag = asset_tag + ([asset_key] * (len(sequences) - temp_sequence_length))

        self.asset_tag = asset_tag

        # create encoder decoder inputs and expected output
        self.encoder_input, self.decoder_input, self.expected_output = [], [], []
        self.sos_token = -torch.ones(len(data[asset_key]['Data'][0])).unsqueeze(0)

        for sequence in sequences:
            self.encoder_input.append(torch.tensor(sequence[:encoder_input_length], dtype=torch.float)[:, layer_features['encoder_features']])
            self.decoder_input.append(torch.tile(self.sos_token, (prediction_length, 1))[:, layer_features['decoder_features']].float())
            self.expected_output.append(torch.tensor(sequence[encoder_input_length:sequence_length], dtype=torch.float)[:, layer_features['decoder_features']])

        if dataset == 'train':
            idx_buy_yes = []
            idx_buy_no = []

            for num_feature, feature in enumerate(self.expected_output):
                if feature[0][0] == 1: idx_buy_yes.append(num_feature) # 0 = index of buy_yes
                else: idx_buy_no.append(num_feature)

            keep = random.sample(idx_buy_no, len(idx_buy_yes)) + idx_buy_yes
            self.encoder_input = np.array(self.encoder_input)[keep]
            self.decoder_input = np.array(self.decoder_input)[keep]
            self.expected_output = np.array(self.expected_output)[keep]
            self.asset_tag = np.array(self.asset_tag)[keep]
           

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        encoder_input = self.encoder_input[idx]
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        asset_tag = self.asset_tag[idx]
        return encoder_input, decoder_input, expected_output, asset_tag




class EvaluationDataset(CustomDataset):
    def __init__(self, data, layer_features, encoder_input_length=50, prediction_length=1, shift=1):
        sequence_length = encoder_input_length + prediction_length
        asset_tag = []
        sequences = []
        timestamps = []

        for asset_key, asset in data.items():
            for num_sequence in range(0, len(asset['Data'][0]) // sequence_length, min(shift, sequence_length)):
                temp_sequence_length = len(sequences)
                sequences = sequences + [shifted_copy[num_sequence:num_sequence + sequence_length] for shifted_copy in asset['Data']]
                timestamps = timestamps + [shifted_copy[num_sequence:num_sequence + sequence_length] for shifted_copy in asset['Datetime']]
                asset_tag = asset_tag + ([asset_key] * (len(sequences) - temp_sequence_length))
        
        self.asset_tag = asset_tag

        # create encoder decoder inputs and expected output
        self.encoder_input, self.decoder_input, self.eval_target = [], [], []
        self.timestamp_last_candle_encoder = []
        self.sos_token = -torch.ones(len(data[asset_key]['Data'][0])).unsqueeze(0)

        for sequence in sequences:
            self.encoder_input.append(torch.tensor(sequence[:encoder_input_length], dtype=torch.float)[:, layer_features['encoder_features']])
            self.decoder_input.append(torch.tile(self.sos_token, (prediction_length, 1))[:, layer_features['decoder_features']].float())
            self.eval_target.append(torch.tensor(sequence[encoder_input_length:sequence_length], dtype=torch.float)[:, layer_features['encoder_features']])
        
        for timestamp in timestamps:
            self.timestamp_last_candle_encoder.append(timestamp[encoder_input_length - 1: encoder_input_length][0])

        

        
    def __getitem__(self, idx):
        encoder_input = self.encoder_input[idx]
        decoder_input = self.decoder_input[idx]
        eval_target = self.eval_target[idx]
        asset_tag = self.asset_tag[idx]
        timestamp_last_candle_encoder = self.timestamp_last_candle_encoder[idx]

        return encoder_input, decoder_input, eval_target, asset_tag, timestamp_last_candle_encoder

