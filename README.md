##### potem dodam opis nie chce mi sie pozdro

## Installation

To install the Quantum State Deep Learning repository use

```bash
git clone https://github.com/Askartan/qsdl
```

### Dependencies

Its recommended to use `uv` as it really speeds up the installation process  

```bash
uv sync
```

Otherwise use

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Quick Overview

### Data Generation

To generate quantum states data, use

```bash
python generate_dataset.py --n-per-class 50 --out data --noise --seed 42  
```
- `--n-per-class`: How many samples per class (7*n for all) 
- `--out`: Folder in which the data will be saved
- `--noise`: Leave if you want random noise applied to samples
- `--seed`: Randomness seed


To visualize generated state data, use

```bash
TODO - nie dziala na razie 
```

### Training

To train the model on generated data use the jupyter notebook in `qsdl/cnn/cnn.ipynb`

I'll add the module script later