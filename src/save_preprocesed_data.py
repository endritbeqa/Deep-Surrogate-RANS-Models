import os
import numpy as np
from src.data import dataset
from torch.utils.data import DataLoader
import config




class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.output_dir = config.output_dir
        self.train_dataset = dataset.Airfoil_Dataset(config, mode='train')
        self.val_dataset = dataset.Airfoil_Dataset(config, mode='validation')
        self.train_dataloader = DataLoader(self.train_dataset, config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, config.batch_size, shuffle=True)


    def train_model(self):
        output_dir_train = "/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty_preprocessed/train"
        output_dir_validation = "/home/blin/PycharmProjects/Thesis/src/data_res_32_uncertainty_preprocessed/validation"



        for data, label in self.train_dataloader:
            save_path = os.path.join(output_dir_train, label[0])
            data = np.squeeze(data, axis=0)
            np.savez(save_path, a=data)

        for data, label in self.val_dataloader:
            save_path = os.path.join(output_dir_validation, label[0])
            data = np.squeeze(data, axis=0)
            np.savez(save_path, a=data)





if __name__ == '__main__':
    train_config = config.get_config()
    trainer = Trainer(train_config)
    trainer.train_model()
