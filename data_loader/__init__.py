from data_loader.datasets import steel
from torch.utils.data import DataLoader


def make_data_loader(dataset, data_dir, batch_size, shuffle=True, num_workers=12, training=True, **kwargs):
    if dataset == 'steel':
        print(f'dataset {dataset}, num_workers {num_workers}')
        train_set = steel.SteelSegmentation(data_dir=data_dir, split='train', **kwargs)
        val_set = steel.SteelSegmentation(data_dir=data_dir, split='val', **kwargs)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
