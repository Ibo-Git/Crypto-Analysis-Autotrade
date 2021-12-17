from json import decoder

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, encoder_input_length=50, prediction_length=1):
        
        # split entire history into sequences
        sequence_length = encoder_input_length + prediction_length
        asset_tag = []
        sequences = []

        for asset_key, asset in data.items():
            temp_sequence_length = len(sequences)
            sequences = sequences + [asset[n:n + sequence_length] for n in range(len(asset) - sequence_length)]
            asset_tag = asset_tag + ([asset_key] * (len(sequences) - temp_sequence_length))

        self.asset_tag = asset_tag

        # create encoder decoder inputs and expected output
        self.encoder_input, self.decoder_input, self.expected_output = [], [], []
        self.sos_token = -torch.ones(len(data[asset_key][0])).unsqueeze(0)

        for sequence in sequences:
            self.encoder_input.append(torch.tensor(sequence[:encoder_input_length]))
            self.decoder_input.append(torch.cat((self.sos_token, torch.tensor(sequence[encoder_input_length:sequence_length-1]))))
            self.expected_output.append(torch.tensor(sequence[encoder_input_length:sequence_length]))


    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        encoder_input = self.encoder_input[idx]
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        asset_tag = self.asset_tag[idx]
        return encoder_input.double(), decoder_input.double(), expected_output.double(), asset_tag



