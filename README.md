# Uncertainty Aware Flow Prediction

This repository contains the code for training and testing different deep learning models predicting the uncertainty distribution of the simulations. 


## Datasets 

The Dataset consists of 5000 simulations of airfoils in the [UIUC database](https://m-selig.ae.illinois.edu/ads/coord_database.html). Each simulation has 25 random snapshot taken between timestep 2500 and 3500 of the simulation.
The data contains input(`x-velocity`, `y-velocity`, `binary mask`) and target(`velocity field(x,y)`, `pressure field`).   

## Models

1. **Hierarchical VAE:** U-net architecture using SwinV2 blocks in the encoder and decoder.

2. **Diffusion ViT(DiT):** ViT encoder with a CNN or MLP decoder using diffusion process.

3. **Diffusion SwinV2(Swin):** SwinV2 encoder with a CNN or MLP decoder using diffusion process.

4. **Diffusion FactFormer(FactFormer):** FactFormer encoder with a CNN or MLP decoder using diffusion process.

## Installation 
Create a pip virtual environment and install the packages in the requirements.txt file.
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Results




## Usage 

### Model Selection 
To train the existing architectures in the models folder change the ***config.model*** field in the `src/config.py` file to the desired 
model name.  
In order to change the model structure itself, go to the config file of the model itself found in the models folder (e.g `src/models/swin/Config_Unet_Swin.py`).

### Train setup 
To change the train setup itself(batch size, number of epoch, dataset etc.) edit the `src/train_config.py` file.
Then run `python train.py` to start the training loop. 

To run multiple training runs simultaneously edit the `src/train_config.py` get_config_parametrized function

### Hyperparameter search
To perform hyperparameter search run `python hyperparameter_search.py`.

### Train your own model 
To train your own model create a new folder in `src/models` and add it to the switch statement in the `src/models/`

## Data Generation 

To generate a new dataset edit the `data_generation/config.py` file according to your needs.
OpenFOAM simulation parameters like number of iterations, timesteps saved, resolution etc. can be changed.
Edit the `config.num_snapshots` field and `config.save_timestep` ranges to select how many and in what time of the simulation snapshots will be saved.
The dataset is generated in parallel, so you can select the number of workers. Each simulation is wrapped in an individual thread to prevent hanging simulations or gmsh errors stopping the dataset generation.
A timeout can be set for converting the .dat file to a mesh, mesh to OpenFOAM and the simulation itself.
Datasets at different resolutions can be generated sequentially but this is still not tested.