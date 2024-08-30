import numpy as np

from src.models.swin import  U_net_SwinV2, Config_UNet_Swin
from src.data import naca_dataset
import tensorflow_datasets as tfds
import config
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.model_config = Config_UNet_Swin.get_config()
        self.model = U_net_SwinV2.U_NET_Swin(self.model_config)
        self.output_dir = config.output_dir
        self.train_dataset = naca_dataset.get_data_from_tfds(self.config, mode='train')


    def train_model(self):
        P_max = float('-inf')
        P_min = float('inf')
        X_max = float('-inf')
        X_min = float('inf')
        Y_max = float('-inf')
        Y_min = float('inf')

        for batch in tfds.as_numpy(self.train_dataset):
            inputs ,targets, label = batch['encoder'], batch['decoder'], batch['label']
            inputs, targets = np.squeeze(inputs, axis =0), np.squeeze(targets, axis=0)
            inputs, targets = np.transpose(inputs,(2,0,1)), np.transpose(targets,(2,0,1))

            boundary = inputs[2]
            boundary[boundary != 0] = 1
            boundary = boundary.flatten().astype(bool)
            c, h, w = targets.shape

            targets = targets.reshape((c, h * w))
            fields = targets[np.tile(boundary, (3, 1))]
            p_max , p_min = np.max(fields[0]), np.min(fields[0])
            x_max, x_min = np.max(fields[1]), np.min(fields[1])
            y_max, y_min = np.max(fields[2]), np.min(fields[2])

            if p_max > P_max: P_max = p_max
            if p_min < P_min: P_min = p_min

            if x_max > X_max: X_max = x_max
            if x_min < X_min: X_min = x_min

            if y_max > Y_max: Y_max = y_max
            if y_min < Y_min: Y_min = y_min

        with open('results.txt', 'w') as file:
            file.write('p_max:{}, p_min:{}, x_max:{}, x_min:{}, y_max:{}, y_min:{},'.format(P_max,P_min,X_max, X_min, Y_max, Y_min))





if __name__ == '__main__':
    train_config = config.get_config()
    trainer = Trainer(train_config)
    trainer.train_model()