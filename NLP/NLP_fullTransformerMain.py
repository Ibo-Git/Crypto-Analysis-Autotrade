
import math
import os
import pathlib
import pickle

import youtokentome as yttm
from torch.utils.data import DataLoader

from NLP_customDataset import CustomDataset, TokenIDX
from NLP_fullTransformerModel import Trainer


class DataProcessing():
    def __init__(self, param):
        self.param = param
        self.vocab_size = param.params['vocab_size']
        self.datapath = param.datapath
        self.initialize_savepath(
            bpe_modename = str(self.vocab_size) + '_bpe.model', # bpemodel name: 'bpe_200.model
            dataset_name = param.dataset_name # name of dataset to create folder: 'savedFiles\\trump
        )

        self.read_dataset(self.datapath)
        

    def initialize_savepath(self, bpe_modename, dataset_name):
        # create save path if it doesn't exist
        if not os.path.isdir('savedFiles'):
            os.mkdir('savedFiles')

        if os.path.isdir(os.path.join('savedFiles', dataset_name)):
            savepath = os.path.join('savedFiles', dataset_name)
        else:
            os.mkdir(os.path.join('savedFiles', dataset_name))
            savepath = os.path.join('savedFiles', dataset_name)
        
        # create files to save model and encoded files
        self.savepath = savepath
        self.save_filepath = os.path.join(savepath, str(self.vocab_size) + '_encoded_files.pkl' )
        self.bpe_modelpath = os.path.join(savepath, bpe_modename)

        # set state depending on whether model name already exists or not
        if os.path.isfile(self.bpe_modelpath):
            self.state = 'load'
        else:
            self.state = 'save'


    def read_dataset(self, datapath):
        if pathlib.Path(os.path.join(datapath, 'lowercase.txt')).is_file():
             with open(os.path.join(datapath, 'lowercase.txt'), 'r', encoding="UTF-8") as file:
                self.text = file.read()
        else:
            current_path = pathlib.Path().absolute()
            files_names = os.listdir(os.path.join(current_path, datapath))
            file_all = []

            for file_name in files_names:
                with open(os.path.join(current_path, datapath, file_name), 'r', encoding="UTF-8") as file:
                    file = file.read().replace('\n', '')
                file_all.append(file)
            
            file_all = ''.join(file_all)
            self.text = file_all.lower()

            f = open(os.path.join(datapath, "lowercase.txt"), "w", encoding="UTF-8")
            f.write(self.text)
            f.close()
    

    def data_preprocessing(self):
        # train bpe on file and save
        if self.state == 'save':
            yttm.BPE.train(
                data = os.path.join(self.datapath, 'lowercase.txt'),
                vocab_size = self.vocab_size,
                model = self.bpe_modelpath,
                pad_id = TokenIDX.PAD_IDX,
                unk_id = TokenIDX.UNK_IDX,
                bos_id = TokenIDX.SOS_IDX,
                eos_id = TokenIDX.EOS_IDX
            )

            self.bpe = yttm.BPE(model=self.bpe_modelpath)
            word_encoded = self.bpe.encode(self.text, output_type=yttm.OutputType.SUBWORD)
            index_encoded = self.bpe.encode(self.text, output_type=yttm.OutputType.ID)

            with open(self.save_filepath, 'wb') as file:
                pickle.dump([word_encoded, index_encoded], file)

        # load bpe model and encoded files
        elif self.state == 'load':
            self.bpe = yttm.BPE(model=self.bpe_modelpath)

            with open(self.save_filepath, 'rb') as file:
                word_encoded, index_encoded = pickle.load(file)

        return index_encoded


    def create_dataloader(self, ID_encoded):
        train_sequences = ID_encoded[0:math.floor(len(ID_encoded) * self.param.split_percent)]
        val_sequences = ID_encoded[math.floor(len(ID_encoded) * self.param.split_percent):]
        train_ds = CustomDataset(train_sequences, encoder_input_length=self.param.encoder_input_length, prediction_length=self.param.prediction_length)
        val_ds = CustomDataset(val_sequences, encoder_input_length=self.param.encoder_input_length, prediction_length=self.param.prediction_length)
        train_dl = DataLoader(train_ds, batch_size=self.param.batch_size['training'], shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
        val_dl = DataLoader(val_ds, batch_size=self.param.batch_size['validation'], shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

        return train_dl, val_dl



class InitializeParameters():
    def __init__(self):
        self.split_percent = 0.8
        self.encoder_input_length = 50
        self.prediction_length = 20
        self.batch_size = {
            'training': 64, 
            'validation': 256, 
        }

        # Hyperparameters
        self.params = {
            # Model
            'vocab_size': 200,
            'encoder_input_length': self.encoder_input_length,
            'n_heads': 2,
            'd_model': 512,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dropout': 0,
            # Optim
            'optim_name': 'Adam',
            'optim_lr': 0.0001,
            # Loss
            'loss_name': 'CrossEntropyLoss'
        }

        self.param_adjust_lr = {
            'lr_decay_factor': 5,
            'loss_decay': 0.05,
            'min_lr': 0.0000035
        }

        self.val_set_eval_during_training = True
        self.eval_mode = False
        self.load_model = False
        self.lr_overwrite_for_load = None 

        # paths and save names
        self.model_name = str(self.params['vocab_size']) + '_small-trumpi-net_xD' # save name of transformer model xD
        self.datapath = 'datasets\\trump\\originals' # path to load data, .txt files should be stored in there
        self.dataset_name = 'trump' # name of folder for saving models etc.




def main():
    # init parameters and process text
    param = InitializeParameters()
    processor = DataProcessing(param=param)
    ID_encoded = processor.data_preprocessing()

    # split into training and validation to create dataloaders
    train_dl, val_dl = processor.create_dataloader(ID_encoded)

     # Start Training and / or Evaluation
    if not param.eval_mode:
        # Create model
        if not param.load_model:
            trainer = Trainer.create_trainer(params=param.params, processor=processor)
        else:
            checkpoint = Trainer.load_checkpoint(modelname=param.model_name, processor=processor)
            trainer = Trainer.create_trainer(params=checkpoint, processor=processor)
            trainer.load_training(modelname=param.model_name, processor=processor)
            if param.lr_overwrite_for_load != None: trainer.set_learningrate(param.lr_overwrite_for_load)

        for epoch in range(1000):
            print(f' --- Epoch: {epoch + 1}')
            trainer.perform_epoch(train_dl, 'train', param.param_adjust_lr)
            trainer.save_training(modelname=param.model_name, processor=processor)

            if param.val_set_eval_during_training:
               trainer.perform_epoch(val_dl, 'val')
    else:
        checkpoint = Trainer.load_checkpoint(modelname=param.model_name, processor=processor)
        trainer = Trainer.create_trainer(params=checkpoint, processor=processor)
        trainer.load_training(modelname=param.model_name, processor=processor)
        trainer.perform_epoch(val_dl, 'val')

        breakpoint = None

 
if __name__ == '__main__':
    main()

