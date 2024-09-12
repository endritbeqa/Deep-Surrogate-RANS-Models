import json
import os
import random

import numpy as np
from ml_collections import config_dict
from src import config
from torch.utils.data import Dataset

'''
        Data is saved in a dictionary in the following format
        DATA = {
                'simulation_000001':{
                                    "timesteps" : Num of timesteps of that simulation (e.g. 1000),
                                    "reynolds_number" : (e.g. 10000),
                                    "mach_number" : (e.g. 0.4),
                                    "pressure": ['/path_to_data_folder/simulation_000001/pressure_000001.npz','/path_to_data_folder/simulation_000001/pressure_000002.npz', ..... ,'/path_to_data_folder/simulation_000001/pressure_001000.npz']
                                    "velocity": ['/path_to_data_folder/simulation_000001/velocity_000001.npz','/path_to_data_folder/simulation_000001/velocity_000002.npz', ..... ,'/path_to_data_folder/simulation_000001/velocity_001000.npz']
                                    "density": ['/path_to_data_folder/simulation_000001/density_000001.npz','/path_to_data_folder/simulation_000001/density_000002.npz', ..... ,'/path_to_data_folder/simulation_000001/density_001000.npz']
                                    # obstacle_mask or velocityZ depending on the dataset used
                },
                'simulation_000002':{
                                    "timesteps" : Num of timesteps of that simulation (e.g. 1000),
                                    "reynolds_number" : (e.g. 10000),
                                    "mach_number" : (e.g. 0.3),
                                    "pressure": ['/path_to_data_folder/simulation_000002/pressure_000001.npz','/path_to_data_folder/simulation_000002/pressure_000002.npz', ..... ,'/path_to_data_folder/simulation_000002/pressure_001000.npz']
                                    "velocity": ['/path_to_data_folder/simulation_000002/velocity_000001.npz','/path_to_data_folder/simulation_000002/velocity_000002.npz', ..... ,'/path_to_data_folder/simulation_000002/velocity_001000.npz']
                                    "density": ['/path_to_data_folder/simulation_000002/density_000001.npz','/path_to_data_folder/simulation_000002/density_000002.npz', ..... ,'/path_to_data_folder/simulation_000002/density_001000.npz']
                }
        }

        '''




#TODO test this file for bugs
#TODO create the train and validation split
class Turbulent_Dataset(Dataset):

    def __init__(self, config: config_dict, mode:str):
        self.mode = config.dataset.data_type
        self.data_dir = str(os.path.join(config.data_dir, self.mode)) #intelliSense complains without the str()
        self.batch_size = config.batch_size
        self.simulation_step = config.datast.simulation_step
        self.context_size = config.context_size
        self.normalize = config.data_preprocessing.normalize
        self.epsilon = 1e-8  # constant for numerical stability
        self.data = {}
        self.simulations_names = sorted(os.listdir(self.data_dir))
        for simulation in self.simulations_names:
            json_file = open(os.path.join(self.data_dir, simulation,"src", 'description.json'))
            simulation_config = json.load(json_file)
            self.data[simulation] = { "timesteps": len(simulation_config.get('Drag Coefficient')),
                "pressure": sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("pressure") and f.endswith(".npz")]),
                "velocity": sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("velocity") and f.endswith(".npz")])
                                     }
            if self.mode == 'tra':
                self.data[simulation]['density'] = sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("density") and f.endswith(".npz")])
                self.data[simulation]['reynolds_number']: simulation_config.get('Reynolds Number')
                self.data[simulation]['mach_number']: simulation_config.get('Mach Number')
            elif self.mode == 'iso':
                self.data[simulation]['velocityZ'] = sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("velocityZ") and f.endswith(".npz")])
            elif self.mode == 'inc':
                self.data[simulation]['reynolds_number']: simulation_config.get('Reynolds Number')
                self.data[simulation]['obstacle_mask'] = sorted([f for f in os.listdir(os.path.join(self.data_dir, simulation)) if f.startswith("obstacle_mask") and f.endswith(".npz")])




    def __len__(self):
        return len(self.file_names)

    #TODO implement timestep skipping (e.g only take timesteps(0,2,4,6,8,10))
    def __getitem__(self, idx):
        simulation_name = 'sim_'+str(idx).zfill(6)
        simulation = self.data[simulation_name]
        starting_timestep = random.randint(0,simulation['timesteps']-(self.context_size*self.simulation_step))
        reynolds_number = simulation['reynolds_number']
        mach_number = simulation['mach_number']
        data = []

        for i in range(starting_timestep, starting_timestep+self.context_size, self.simulation_step):
            pressure = np.load(os.path.join(self.data_dir,simulation_name,simulation["pressure"][i]))['arr_0']
            velocity = np.load(os.path.join(self.data_dir,simulation_name,simulation["velocity"][i]))['arr_0']
            fields = None


            if self.mode == 'tra':
                density = np.load(os.path.join(self.data_dir, simulation_name, simulation["density"][i]))['arr_0']
                reynolds = np.full_like(pressure, reynolds_number)
                mach = np.full_like(pressure, mach_number)
                fields = np.concatenate((reynolds, mach, velocity, density, pressure), axis=0)
            elif self.mode == 'iso':
                velocity_Z = np.load(os.path.join(self.data_dir, simulation_name, simulation["velocityZ"][i]))['arr_0']
                fields = np.concatenate((velocity, velocity_Z, pressure), axis=0)
            elif self.mode == 'inc':
                obstacle_mask = np.load(os.path.join(self.data_dir, simulation_name, simulation["obstacle_mask"][i]))['arr_0']
                reynolds = np.full_like(pressure, reynolds_number)
                fields = np.concatenate((reynolds, obstacle_mask,velocity, pressure), axis=0)

            fields = self.preprocess_data(fields)
            data.append(fields)

        data = np.array(data)
        return data


    #TODO implement the preprocessing
    def preprocess_data(self, data) -> list[np.ndarray]:
        c, h, w = data.shape
        normMean = np.array([])
        normStd = np.array([])

        # mean and std statistics from whole dataset for normalization
        #TODO check the values again
        if self.normalize:

            if self.mode == "tra":
                # ORDER (fields): reynolds, mach, velocity(x,y), density, pressure
                normMean = np.array([10000.000000, 0.700000,0.560642, -0.000129, 0.903352, 0.637941], dtype=np.float32)
                normStd = np.array([1, 0.118322, 0.216987, 0.216987, 0.145391, 0.119944, ], dtype=np.float32)

            elif self.mode == "iso":
                # ORDER (fields):velocity(x,y,z), pressure
                normMean = np.array([-0.054618, -0.385225, -0.255757, 0.033446], dtype=np.float32)
                normStd = np.array([0.539194, 0.710318, 0.510352, 0.258235], dtype=np.float32)

            elif self.mode == "inc":
                # ORDER (fields): reynolds, obstacle_mask, velocity(x,y), pressure
                normMean = np.array([550.000000, 0, 0.444969, 0.000299, 0.000586], dtype=np.float32)
                normStd = np.array([262.678467, 1, 0.206128, 0.206128, 0.003942], dtype=np.float32)

            normMean = np.tile(normMean[:, np.newaxis, np.newaxis], (1,h,w))
            normStd = np.tile(normStd[:, np.newaxis, np.newaxis], (1,h,w))

            data = (data - normMean) / normStd

        return data





