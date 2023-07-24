# ARCOO

The code for the paper "Degradation-Resistant Offline Optimization via Accumulative Risk Control".

## Setup
Create the running environment with conda `4.10.3` with Python `3.9.0`: 

```shell
conda create -n arcoo python==3.9
conda activate arcoo
```

Install the requirements for running ARCOO: 

```shell
pip install -r requirements.txt
```

## Offline Optimization Tasks
All the experiments are conducted on 3 discrete offline optimization tasks and 3 continuous ones based on the problems of DNA optimization, drug discovery, material invention, and robotic design from [Design-Bench](https://github.com/brandontrabucco/design-bench). Here we quote a brief introduction from the original page. Refer to the original page for more deatals and usages.
> Design-Bench is a benchmarking framework for solving automatic design problems that involve choosing an input that maximizes a black-box function. This type of optimization is used across scientific and engineering disciplines in ways such as designing proteins and DNA sequences with particular functions, chemical formulas and molecule substructures, the morphology and controllers of robots, and many more applications.


## Run experiments
All the experiments of ARCOO can be find in `experiments.py`. The following command line are availuable to run experiments:

### ARCOO on 6 tasks
```python
python experiments.py tfbind8_arcoo_exp $random_seed
python experiments.py chembl_arcoo_exp $random_seed
python experiments.py utr_arcoo_exp $random_seed
python experiments.py supercond_arcoo_exp $random_seed
python experiments.py ant_arcoo_exp $random_seed
python experiments.py hopper_arcoo_exp $random_seed
```

#### Hyperparameters analyze on initial $m$ and $E_{max}$

```python
python experiments.py hopper_arcoo_exp_hyper_m_emax $random_seed
```

#### Alation study of key componets

```python
python experiments.py hopper_arcoo_exp_alation $random_seed
```

#### Hyperparameters analyze on initial $m$ and $E_{max}$

```python
python experiments.py hopper_arcoo_exp_hyper_m_emax $random_seed
```


## The project structure
```
ARCOO
├─ config
│  └─ 
│     ├─ chembl
│     │  └─ default.json
│     ├─ dkitty
│     │  └─ default.json
│     ├─ hopper
│     │  └─ default.json
│     ├─ supercond
│     │  └─ default.json
│     ├─ tfbind8
│     │  └─ default.json
│     └─ utr
│        └─ default.json
├─ core
│  ├─ data.py
│  ├─ arcoo
│  │  ├─ nets.py
│  │  ├─ optimizer.py
│  │  ├─ trainers.py
│  │  └─ __init__.py
│  ├─ exp_conductor.py
│  ├─ logger.py
│  └─ utils.py
├─ experiments.py
├─ README.md
└─ requirements.txt
```

Notes for the project structure:
- The files in the folder `core` are the main components of the algorithms.
- The files in the folder `config` are the hyperparameter configurations for each task.




