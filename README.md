# Memory Enhanced Spatial-Temporal Graph Convolutional Autoencoder for Human-related Video Anomaly Detection

This is the implementation of the paper "Memory Enhanced Spatial-Temporal Graph Convolutional Autoencoder for Human-related Video Anomaly Detection"
## Dependencies
- Pytoch 1.7.0
- Numpy
- SciPy
- Sklearn
## Dataset
**ShanghaiTech**
please download the data from the following link
Link: [trajectories](https://bit.ly/2TWCxFY)
## Directory Structure
.
├── models       -- Including graph definitions and convolution operators
├── utils        -- Data process and score utils
├── data         -- Dataset directory
├── README.md          
└── train_eval.py     -- Main file for training / inference
## Training
To train a model from scratch you should look up the model's configuration options.
Here is one example:
`python train_eval.py --data_dir {dataset folder} --exp_dir {path to save experiment result}`

