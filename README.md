*"When you can write well, you can think well."* 

--Matt Mullenweg.

# MetaViterbiNet

Python repository for the paper "Meta-ViterbiNet: Online Meta-Learned Viterbi Equalization for Non-Stationary Channels".

Please cite our paper (*link will be here*), if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python-code)
    + [channel](#channel)
    + [detectors](#detectors)
    + [ecc](#ecc)
    + [plotters](#plotters)
    + [trainers](#trainers)
    + [utils](#utils)
    + [config](#config)
  * [resources](#resources)
  * [dir_definitions](#dir-definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This repository implements classical and machine-learning based detectors for a channel with memory of L. We implemented the naive [Viterbi algorithm](https://ieeexplore.ieee.org/document/1054010), as well as the [ViterbiNet](https://ieeexplore.ieee.org/document/8815457) in python. Our method, for incorporating temporal evolution over a sequence of symbols, is referred to as Meta-ViterbiNet. We also implemented a model-free baseline of a windowed-LSTM detector. We explain on the different directories and subdirectories below.

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: encoder, channel and detectors.

### channel 

Includes all relevant channel functions and classes. The class in "channel_dataset.py" implements the main class for aggregating pairs of (transmitted,received) pairs. 
In "channel.py", the ISI AWGN channel is implemented. "channel_estimation.py" is for the calculation of the h values. Lastly, the channel BPSK modulator lies in "channel_modulator.py".

### detectors

The backbone detectors: VA, VNET, LSTM, META_VNET and META_LSTM. The meta and non-meta detectors have slightly different API so they are seperated in the trainer class below. Also, we use VA as the ML detector, thus we assume full knowledge of the CSI. To have a single API across the detectors, the snr and gamma appear in all the approriate forward calls, but are omitted in the code itself. A factory design pattern could have been a better fit here, and is left as future work.

### trainers 

Wrappers for the training and evaluation of the detectors.

The basic trainer class holds most used methods: train, meta-train and evaluation. It is also used for parsing the config.yaml file, preparing the

Each trainer inherets from the basic trainer class, extending it as needed.

### plotting

Plotting of the FER versus SNR, and the FER versus the states. See Figures 4-6 in the paper.

### utils

Extra utils for saving and loading pkls; tail-biting related calculations; and more...

### config

Controls all parameters and hyperparameters in Tables I and II.

## resources

Folder with all codes matrices.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 2060 with driver version 432.00 and CUDA 10.1. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f deep_ensemble.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\deep_ensemble\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
