from json import decoder

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, crypto_data, encoder_input_length=50, prediction_length=3):
        self.sos_token = torch.zeros(4).unsqueeze(0)
        self.eos_token = torch.ones(4).unsqueeze(0)

        # split entire history into sequences
        sequence_length = encoder_input_length + prediction_length
        sequences = [crypto_data[n:n + sequence_length] for n in range(len(crypto_data) - sequence_length)]

        # create encoder decoder inputs and expected output
        encoder_input = []
        decoder_input = []
        expected_output = []

        for sequence in sequences:
            encoder_input.append(sequence[:encoder_input_length])
            decoder_input.append(sequence[encoder_input_length:sequence_length])
            expected_output.append(sequence[encoder_input_length:sequence_length])

        self.encoder_input = []
        self.decoder_input = []
        self.expected_output = [] 

        for idx in range(len(expected_output)):
            self.encoder_input.append([
                torch.tensor(encoder_input[idx]['Open']), 
                torch.tensor(encoder_input[idx]['High']), 
                torch.tensor(encoder_input[idx]['Low']), 
                torch.tensor(encoder_input[idx]['Close'])
            ])

            self.decoder_input.append([
                torch.tensor(decoder_input[idx]['Open']),
                torch.tensor(decoder_input[idx]['High']),
                torch.tensor(decoder_input[idx]['Low']),
                torch.tensor(decoder_input[idx]['Close'])
            ])

            self.expected_output.append([
                torch.tensor(expected_output[idx]['Open']),
                torch.tensor(expected_output[idx]['High']),
                torch.tensor(expected_output[idx]['Low']),
                torch.tensor(expected_output[idx]['Close'])
            ])

           
    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        encoder_input = torch.stack(self.encoder_input[idx]).permute(1, 0)
        decoder_input = torch.cat((self.sos_token, torch.stack(self.decoder_input[idx]).permute(1, 0)))
        expected_output = torch.cat((torch.stack(self.expected_output[idx]).permute(1, 0), self.eos_token))
        return encoder_input, decoder_input, expected_output


