# Uncertainty Aware Flow Prediction

This repository contains the code for training and testing different deep learning models predicting the uncertainty
distribution of the simulations. Idea(incorporate openFoam solver between steps, this would require a graph
representation of the grid and graph neural networks, need a way to easily convert from foam file to python data to
transfer to gpu and run the ml models on that input)

## Datasets

The [Dataset](https://github.com/tum-pbs/Diffusion-based-Flow-Prediction?tab=readme-ov-file) consists of 5000
simulations of airfoils in the [UIUC database](https://m-selig.ae.illinois.edu/ads/coord_database.html). Each simulation
has 25 random snapshot taken between timestep 2500 and 3500 of the simulation.
The data contains input(`x-velocity`, `y-velocity`, `binary mask`) and target(`velocity field(x,y)`, `pressure field`).

## Models

1. **Hierarchical VAE:** U-net architecture using SwinV2 blocks in the encoder and decoder.

2. **Diffusion ViT(DiT):** ViT encoder with a CNN or MLP decoder using diffusion process.

3. **Diffusion SwinV2(Swin):** SwinV2 encoder with a CNN or MLP decoder using diffusion process.

4. **Diffusion FactFormer(FactFormer):** FactFormer encoder with a CNN or MLP decoder using diffusion process.

## Installation

Create a virtual environment and install the packages in the requirements.txt file. `Python3.9` is required.

```shell
python3 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

## Results

## Usage

### Data download

The dataset and instructions can be found [here](https://mediatum.ub.tum.de/1731896). After download edit the flags in
`src/data/preprocess_data.py`(beginning of the file) and run it. Afterwards edit the `train_config.data_dir` field in
`train_config.py` to point to the train_val_split folder created.

### Model Selection

To train the existing architectures in the models folder change the ***config.model_name*** field in the
`src/train_config.py` file to the desired
model name.  
In order to change the model structure itself, go to the config file of the model itself found in the models folder (e.g
`src/models/diffusion/Swin/Config_Swin.py`).

### Train setup

To change the train setup itself(batch size, number of epoch, dataset etc.) edit the `src/train_config.py` file.
Then run `python -m src.train.py` to start the training loop.

To run multiple training runs simultaneously in `src/train_config.py` edit `get_config_parametrized()` function to set
hyperparameters for all training runs.
In the `src/train_multiprocess.py` edit the study_name, model_name, seeds, devices list to run separate training runs.
In the end run `python -m src.train_multiprocess`.

To restart previous training runs edit the checkpoints, data_dir, output_dir, devices list to restart the training and
set the global flag `MODE = 'restart'` in the `src/train_multiprocess.py` file.
In the end run `python -m src.train_multiprocess`.

### Evaluation

To run evaluation on a model edit the `src/evaluation/evaluation_config.py` file and then run
`python -m src.evaluation.evaluate_model`.

Currently there are 3 test:

1. Inter/Extrapolation
2. Sampling Speed
3. Coefficient of Drag

### Hyperparameter search

To perform hyperparameter search run `python hyperparameter_search.py`.

### Train your own model

To train your own model create a new folder in `src/models` and add it to the switch statement in the
`src/models/model_select`. If it is a Diffusion model don't forget to wrap it in the `src/models/Diffuser` class.

## Data Generation

To generate a new dataset edit the `data_generation/config.py` file according to your needs.
OpenFOAM simulation parameters like number of iterations, timesteps saved, resolution etc. can be changed.
Edit the `config.num_snapshots` field and `config.save_timestep` ranges to select how many and in what time of the
simulation snapshots will be saved.
The dataset is generated in parallel, so you can select the number of workers. Each simulation is wrapped in an
individual thread to prevent hanging simulations or gmsh errors stopping the dataset generation.
A timeout can be set for converting the .dat file to a mesh, mesh to OpenFOAM and the simulation itself.
Datasets at different resolutions can be generated sequentially but this is still not tested.