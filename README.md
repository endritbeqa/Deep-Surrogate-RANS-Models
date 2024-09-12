# Turbulent Flow Prediction

This repository contains the code for training and testing different deep learning models predicting turbulent flow. 


## Datasets 

All architectures are evaluated on three carefully designed data sets of varying difficulty:

1. **Incompressible Wake Flow (*Inc*):** This relatively simple data set contains 91 incompressible wake flow trajectories simulated with PhiFlow at Reynolds numbers 100-1000. Each sequence contains 1300 temporal snapshots of pressure and velocity.


2. **Transonic Cylinder Flow (*Tra*):** This data set contains 41 sequences of a two-dimensional transonic cylinder flow simulated with the fluid solver SU2. Each sequence has a different Mach number between 0.5 and 0.9 at Reynolds number 10000, making the flows in the transonic regime highly chaotic. All trajcetories feature 1000 temporal snapshots of pressure, density, and velocity.


3. **Isotropic Turbulence (*Iso*):** This highly complex, underdetermined data set consists of 1000 two-dimensional sequences sliced from three-dimensional, isotropic turbulence simulations. The flows were computed with DNS and the raw data is provided by the Johns Hopkins Turbulence Database. Each sequence contains 1000 temporal snapshots of pressure and velocity, including the velocity z-component.


4. **Airfoil Turbulence Dataset (*Air*):** This dataset has yet to be generated with the data_generation folder 

Further information about the datasets and the previous work by the [Thuerey group](https://ge.in.tum.de/) can be found in the [github repository](https://github.com/tum-pbs/autoreg-pde-diffusion) and at the [project website](https://ge.in.tum.de/publications/2023-acdm-kohl/).


## Installation 
Create a pip virtual environment and install the packages in the requirements.txt file.
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Folder Structure 
#TODO automate this with github actions and add shell colors. 
```
.
├── data_generation
│       ├── airfoil_database_test
│       ├── airfoil_database_train
│       ├── config.py
│       ├── dataGeneration.py
│       ├── download_airfoils.sh
│       ├── gen_data.sh
│       ├── main.py
│       ├── OpenFOAM
│       ├── simFunctions.py
│       └── utils.py
├── LICENSE
├── README.md
├── req.txt
└── src
    ├── config.py
    ├── data
    ├── hyperparameter_search.py
    ├── loss.py
    ├── models
    ├── train.py
    └── utils.py
```

## Usage 

### Model Selection 
To train the existing architectures in the models folder change the ***config.model*** field in the `src/config.py` file to the desired 
model name.  
In order to change the model structure itself, go to the config file of the model itself found in the models folder (e.g `src/models/swin/Config_Unet_Swin.py`).

### Loss function selection
To select loss a loss function edit the ***config.loss_function*** field in the `src/config.py`. You can also use the sum of multiple loss functions at once by just typing the names in form of a list (e.g you are training a VAE version of the model and need also the KL divergence term in the loss )

### Train setup 
To change the train setup itself(batch size, number of epoch, dataset etc.) edit the `src/config.py` file. then run `python train.py` to start the training loop

### Dataset selection 
To change the dataset used during training edit the ***config.dataset.data_type*** field in the `src/config.py` file.

### Hyperparameter search
To perform hyperparameter search edit the `src/hyperparameter_search.py` file and run `python hyperparameter_search.py`

### Train your own model 
To train your own model create a new folder in `src/models` and add it to the switch statement in the `src/models/`

## Data Generation 

