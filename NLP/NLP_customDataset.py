import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TokenIDX():
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3


class CustomDataset(Dataset):
    def __init__(self, data, encoder_input_length=50, prediction_length=5):
        
        # split entire text into sequences
        sequence_length = encoder_input_length + prediction_length
        sequences = [data[n:n + sequence_length] for n in range(0, len(data) - sequence_length, sequence_length)]

        # create encoder decoder inputs and expected output
        self.encoder_input, self.decoder_input, self.expected_output = [], [], []

        print('Create transformer inputs...')
        for sequence in tqdm(sequences):
            self.encoder_input.append(torch.tensor(sequence[:encoder_input_length]))
            self.decoder_input.append(torch.cat((torch.tensor(TokenIDX.SOS_IDX).unsqueeze(0), torch.tensor(sequence[encoder_input_length:sequence_length-1]))))
            self.expected_output.append(torch.tensor(sequence[encoder_input_length:sequence_length]))


    def __len__(self):
        return len(self.encoder_input)


    def __getitem__(self, idx):
        encoder_input = self.encoder_input[idx]
        decoder_input = self.decoder_input[idx]
        expected_output = self.expected_output[idx]
        return encoder_input, decoder_input, expected_output



