from json import decoder
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, crypto_data, sequence_length=50):
        sequences = [crypto_data[n:n + sequence_length] for n in range(len(crypto_data) - sequence_length)]
        decoder_input = sequences[0:-2]
        expected_output = sequences[1:-1]

        self.expected_output = [] 
        self.decoder_input = []

        for idx in range(len(expected_output)):
            self.expected_output.append([
                torch.tensor(expected_output[idx]['Open']), 
                torch.tensor(expected_output[idx]['High']), 
                torch.tensor(expected_output[idx]['Low']), 
                torch.tensor(expected_output[idx]['Close'])
            ])

            self.decoder_input.append([
                torch.tensor(decoder_input[idx]['Open']), 
                torch.tensor(decoder_input[idx]['High']), 
                torch.tensor(decoder_input[idx]['Low']), 
                torch.tensor(decoder_input[idx]['Close'])
            ])
        
    def __len__(self):
        return len(self.decoder_input)

    def __getitem__(self, idx):
        decoder_input = torch.stack(self.decoder_input[idx]).permute(1, 0)
        expected_output = torch.stack(self.expected_output[idx]).permute(1, 0)
        return decoder_input, expected_output


