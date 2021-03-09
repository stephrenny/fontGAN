# fontGAN (WIP)
Generative Adversarial Network to edit text in images while keeping the font and remaining content consistent

# Setup

## Create environment
Create the conda environment with 

`conda env create -f environment.yml`

Activate the envionment with 

`conda activate fonts`

## Gather data
To pull the data and set up the project, run

`python setup.py`

# Training
To train the neural network, simply activate the environment and run 

`python train.py -name {my_model_name}`

Training metrics will be logged using tensorboard. To view, run 

`tensorboard --logdir save --port 5678`

And connect to the specified port using ssh

# View Dataset Exmaples

# View Model Output
