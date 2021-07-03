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
* pyyaml==5.3
* pillow==7.0.0
* gym==0.17.2

Please note: PyTorch can be installed via different package managers (e.g. conda), with and without GPU-support etc.:

#### Linux

```bash
pip install torch torchvision
```

#### Windows
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

Alternatively, you can install all requirements in *requirements.txt*:

```bash
pip install -r requirements.txt
```

## Usage

Download and unzip or clone repository. Run the following command

```bash
cd run
python run.py -h
```

to get a brief overview of all possible options:

```
usage: Attention-driven learning of Temporal Abstractions in Reinforcement Learning
       [-h] [-m {darqn,cead}] [-a {concat,general,dot}] [-e ENVIRONMENT] [-o OUTPUT]

Parse arguments for run script

optional arguments:
  -h, --help            show this help message and exit
  -m {darqn,cead}, --model {darqn,cead}
                        model type. defines if darqn or cead model is used.
  -a {concat,general,dot}, --alignment {concat,general,dot}
                        alignment method. defines which alignment method is used for computing attention
                        weights.
  -e ENVIRONMENT, --environment ENVIRONMENT
                        environment. defines which environment is used for training. note: environments must
                        start with a capital letter, e.g. -e Pong
  -o OUTPUT, --output OUTPUT
                        output ID. running the script for the first time generates an output directory in
                        the root directory of this project. the -a argument defines the ID of the run, which
                        ismandatory if the training is interrupted and later continued for some reason.

```
