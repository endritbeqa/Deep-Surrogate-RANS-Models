import json
import os
import random

import numpy as np
from ml_collections import config_dict
from src import config
from torch.utils.data import Dataset

#TODO test this file for bugs
class Turbulent_Dataset(Dataset):

    def __init__(self, config: config_dict):
        #self.data_dir = os.path.join(config.data_dir, mode)
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.context_size = config.context_size
        self.fixedAirfoilNormalization = config.data_preprocessing.fixedAirfoilNormalization
        self.makeDimLess = config.data_preprocessing.makeDimLess
        self.removePOffset = config.data_preprocessing.removePOffset
        self.data = {}
        self.simulations_names = sorted(os.listdir(self.data_dir))
        for simulation in self.simulations_names:
            json_file = open(os.path.join(self.data_dir, simulation,"src", 'description.json'))
            simulation_config = json.load(json_file)
            self.data[simulation] = { "timesteps": len(simulation_config.get('Drag Coefficient')),
                                      "reynolds_number": simulation_config.get('Reynolds Number'),
                                      "mach_number": simulation_config.get('Mach Number'),
                "pressure": sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("pressure") and f.endswith(".npz")]),
                "velocity": sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("velocity") and f.endswith(".npz")])
                                     }
            try:
                self.data[simulation]['density'] = sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("density") and f.endswith(".npz")])
            except:
                print("No density data provided")
        self.epsilon =1e-8 #constant for numerical stability


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        simulation_name = 'sim_'+str(idx).zfill(6)
        simulation = self.data[simulation_name]
        starting_timestep = random.randint(0,simulation['timesteps'])
        reynolds_number = simulation['reynolds_number']
        mach_number = simulation['mach_number']
        data = []

        for i in range(starting_timestep, starting_timestep+self.context_size):
            pressure = np.load(os.path.join(self.data_dir,simulation_name,simulation["pressure"][i]))['arr_0']
            velocity = np.load(os.path.join(self.data_dir,simulation_name,simulation["velocity"][i]))['arr_0']
            fields = np.concatenate((pressure,velocity), axis=0)

            try:
                density = np.load(os.path.join(self.data_dir,simulation_name,simulation["density"][i]))['arr_0']
                fields = np.concatenate((fields,density), axis=0)
            except:
                print("No density field provided")

            #fields = self.preprocess_data(fields)
            data.append(fields)

        data = np.array(data)
        return data, reynolds_number, mach_number



    def preprocess_data(self, data) -> list[np.ndarray]:

        if not any((self.removePOffset, self.makeDimLess, self.fixedAirfoilNormalization)):
            return data

        boundary = data[2].flatten().astype(bool)
        num_field_elements = np.sum(boundary)
        c, h, w = data.shape

        data = data.reshape((c, h * w))
        fields = data[np.tile(boundary, (6, 1))]
        fields = fields.reshape((6, num_field_elements))
        p_mean = np.mean(fields[3])
        v_norm = (np.max(np.abs(fields[0])) ** 2 + np.max(np.abs(fields[1])) ** 2) ** 0.5

        if self.removePOffset:
            data[3][boundary] -= p_mean
            data[3][boundary][data[3][boundary] == 0] = self.epsilon

        if self.makeDimLess:
            data[3][boundary] /= (v_norm ** 2 + self.epsilon)
            data[4][boundary] /= (v_norm + self.epsilon)
            data[5][boundary] /= (v_norm + self.epsilon)

        if self.fixedAirfoilNormalization:
            # hard coded maxima , inputs dont change
            max_inputs_0 = 100.
            max_inputs_1 = 38.12
            max_inputs_2 = 1.0

            # targets depend on normalization
            if self.makeDimLess:
                max_targets_0 = 4.65
                max_targets_1 = 2.04
                max_targets_2 = 2.37

            else:  # full range
                max_targets_0 = 40000.
                max_targets_1 = 200.
                max_targets_2 = 216.

        else:
            max_inputs_0 = np.max(fields[0]) if np.max(fields[0]) !=0 else self.epsilon
            max_inputs_1 = np.max(fields[1]) if np.max(fields[1]) !=0 else self.epsilon

            max_targets_0 = np.max(fields[3]) if np.max(fields[3]) !=0 else self.epsilon
            max_targets_1 = np.max(fields[4]) if np.max(fields[4]) !=0 else self.epsilon
            max_targets_2 = np.max(fields[5]) if np.max(fields[5]) !=0 else self.epsilon

        data[0][boundary] *= (1.0 / max_inputs_0)
        data[1][boundary] *= (1.0 / max_inputs_1)

        data[0][boundary] *= (1.0 / max_targets_0)
        data[1][boundary] *= (1.0 / max_targets_1)
        data[2][boundary] *= (1.0 / max_targets_2)

        data = data.reshape((c, h, w))

        return data

if __name__ == '__main__':
    configgg = config.get_config()

    train_dataset = Turbulent_Dataset(configgg)

    for data, rey, mach in train_dataset:
        print("hola")

