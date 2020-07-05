# Attention-driven Learning of Temporal Abstractions in Reinforcement Learning

Code repository for master thesis 'Attention-driven Learning of Temporal Abstractions in Reinforcement Learning'.

## Getting Started

The code for this thesis is completely written in Python. We use Python 3.7

### Prerequisites

The following packages are required to run the code in this repository:

* numpy==1.18.1
* matplotlib==3.1.3
* pytorch==1.5.0
* torchvision==0.6.0
* yaml==0.1.7
* gym==0.17.2

Please note: PyTorch can be installed via different package managers (e.g. conda), with and without GPU-support etc.:

```bash
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```

For further instructions please visit [PyTorch](https://pytorch.org/)



For usage of Atari gym environments in Windows you can install [atari-py](https://github.com/openai/atari-py)
- atari-py==1.2.1

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages:

```bash
pip install numpy==1.18.1
...
```

Alternatively, you can install all requirements in *requirements.txt*

```bash
pip install -f requirements.txt
```

## Usage

Download and unzip or clone repository. Run the following command 

```bash
python run/run.py -h
```

to get a brief overview of all possible options

```bash

usage: Attention-driven learning of Temporal Abstractions for Reinforcement Learning
       [-h] [-l LSTM_FIRST] [-c CONFIG] [-env {Breakout-v0,another}]
       [-o OUTPUT] [-m {train,eval}]

Parse arguments for run script

optional arguments:
  -h, --help            show this help message and exit
  -l LSTM_FIRST, --lstm-first LSTM_FIRST
                        specifies network architecture. -l or --lstm-first
                        followed by 'True'puts encoder / decoder LSTMs in the
                        first place, followed by convolutional layers if not
                        provided, 'False' is used
  -c CONFIG, --config CONFIG
                        parse config file as '*.yaml' if not provided, default
                        config is used.
  -env {Breakout-v0,another}, --environment {Breakout-v0,another}
                        used for game selection. if not provided,
                        'Breakout-v0' is used
  -o OUTPUT, --output OUTPUT
                        output directory. ensure you have permissions to write
                        to this directory! if not provided, default-directory
                        '/output' is used
  -m {train,eval}, --mode {train,eval}
                        sets mode for run script. 'train' is used for training
                        mode, 'eval' is used for evaluation mode. if not
                        provided, 'train' is used
```

## Contributing

 TODO

## License
None
