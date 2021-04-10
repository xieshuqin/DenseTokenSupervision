from torch.utils.data import DataLoader
from .haa500 import HAA500Dataset, build_haa500_dataset
from .finegym import FineGym, build_finegym_dataset

DATASETS = {'haa500': build_haa500_dataset, 'finegym': build_finegym_dataset}


def build_dataloader(dataset_name, split, batch_size, **dataset_kwargs):
    assert (split in ['train', 'val', 'test']), f'Unsupported split {split}'
    dataset = DATASETS[dataset_name](split=split, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(
        split == 'train'), num_workers=4, pin_memory=True)
    return dataloader
