## Prepare imagenet data
Download ImageNet 2012 tar files for train, val and development tools.

Modify datapath in `train_patch.py` to your download dir. Then proceed with commands below. 


## Finetune pretrained video transformer on finegym:
```python main.py```

## Finetune pretrained ViT on imagenet
### Training with patch supervision
```python train_patch.py ```
### Training without patch supervision
```python train_patch.py --w_patch 0.```

## Finetune deit on imagenet
### Training with soft label
```python train_deit.py```

### Training with hard label
```python train_deit.py --hard_dist```


## Train from scratch on ImageNet data
### Patch supervision
```python train_patch.py --from_scratch```
### No patch supervision
```python train_patch.py --from_scratch```

## Places365
Modify the path to dataset accordingly. 
If you want to download the dataset, pass `--download` to command line.

### Train with patch supervision and download dataset
```python train_patch.py --dataset places365 --lr 6.25e-5 --w_patch 1.0 --download```

### Train without patch supervision and doesn't download dataset
```python train_patch.py --dataset places365 --lr 6.25e-5 --w_patch 0.0```