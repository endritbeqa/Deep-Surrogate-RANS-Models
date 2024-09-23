import os
import numpy as np
from src.data import dataset
from torch.utils.data import DataLoader
import src.config as config




class Preprocesser(object):
    def __init__(self, config):
        self.config = config
        self.output_dir = config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, config.batch_size)
        self.val_dataloader = DataLoader(self.val_dataset, config.batch_size)


    def preprocess(self):
        output_dir_train = os.path.join(self.output_dir, "train")
        output_dir_validation = os.path.join(self.output_dir, "validation")
        os.makedirs(output_dir_train)
        os.makedirs(output_dir_validation)

        train_len = len(self.train_dataset)
        val_len = len(self.val_dataset)

        for idx, (input, target, label) in enumerate(self.train_dataloader):
            print('Train file: {}/{}'.format(idx, train_len))
            save_path = os.path.join(output_dir_train, label[0])
            input, target = input.numpy().squeeze(), target.numpy().squeeze()
            data = np.concatenate((input, target), axis=0)
            np.savez(save_path, a=data)

        for idx, (input, target, label) in enumerate(self.val_dataloader):
            print('Validation file: {}/{}'.format(idx, val_len))
            save_path = os.path.join(output_dir_validation, label[0])
            input, target = input.numpy().squeeze(), target.numpy().squeeze()
            data = np.concatenate((input, target), axis=0)
            np.savez(save_path, a=data)

if __name__ == '__main__':
    train_config = config.get_config()
    trainer = Preprocesser(train_config)
    trainer.preprocess()