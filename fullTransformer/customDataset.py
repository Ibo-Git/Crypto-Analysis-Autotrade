from json import decoder

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, layer_features, encoder_input_length=50, prediction_length=1, shift=1):
        
        # split entire history into sequences
        sequence_length = encoder_input_length + prediction_length
        asset_tag = []
        sequences = []

        for asset_key, asset in data.items():
            for asset_overlap in asset:
                temp_sequence_length = len(sequences)
                sequences = sequences + [asset_overlap[n:n + sequence_length] for n in range(0, len(asset_overlap) - sequence_length, min(shift, sequence_length))]
                asset_tag = asset_tag + ([asset_key] * (len(sequences) - temp_sequence_length))

        self.asset_tag = asset_tag

        # create encoder decoder inputs and expected output
        self.encoder_input, self.decoder_input, self.expected_output = [], [], []
        self.sos_token = -torch.ones(len(data[asset_key][0])).unsqueeze(0)

        for sequence in sequences:
            self.encoder_input.append(torch.tensor(sequence[:encoder_input_length], dtype=torch.float)[:, layer_features['encoder_features']])
            self.decoder_input.append(torch.tile(self.sos_token, (prediction_length, 1))[:, layer_features['decoder_features']].float())
            # self.decoder_input.append(torch.cat((self.sos_token, torch.tensor(sequence[encoder_input_length:sequence_length-1]))).double())
            self.expected_output.append(torch.tensor(sequence[encoder_input_length:sequence_length], dtype=torch.float)[:, layer_features['decoder_features']])


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

        for asset_key, asset in data.items():
            for num_sequence in range(0, len(asset[0]) // sequence_length, min(shift, sequence_length)):
                temp_sequence_length = len(sequences)
                sequences = sequences + [shifted_copy[num_sequence:num_sequence + sequence_length] for shifted_copy in asset]
                asset_tag = asset_tag + ([asset_key] * (len(sequences) - temp_sequence_length))
        
        self.asset_tag = asset_tag

        # create encoder decoder inputs and expected output
        self.encoder_input, self.decoder_input, self.eval_target = [], [], []
        self.sos_token = -torch.ones(len(list(data.values())[0])).unsqueeze(0)

        for sequence in sequences:
            self.encoder_input.append(torch.tensor(sequence[:encoder_input_length], dtype=torch.float)[:, layer_features['encoder_features']])
            self.decoder_input.append(torch.tile(self.sos_token, (prediction_length, 1))[:, layer_features['decoder_features']].float())
            self.eval_target.append(torch.tensor(sequence[encoder_input_length:sequence_length], dtype=torch.float)[:, layer_features['encoder_features']])
        

        
    def __getitem__(self, idx):
        encoder_input = self.encoder_input[idx]
        decoder_input = self.decoder_input[idx]
        eval_target = self.eval_target[idx]
        asset_tag = self.asset_tag[idx]
        return encoder_input, decoder_input, eval_target, asset_tag

