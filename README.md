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