## Install dependencies

```
conda create -n ucc python=3.10
pip install torch torchvision lightning albumentations
conda activate ucc
```

## Unzip warmup.zip in data folder so it has the following structure

```
data/warmup/img/train
data/warmup/img/valid
data/warmup/ann/train
data/warmup/ann/valid
```

## Training
```
python train.py
```