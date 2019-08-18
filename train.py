import argparse
import collections
import torch
import os
import numpy as np
# import data_loader.data_loaders as module_data
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.calculate_weights import calculate_weigths_labels
# from trainer import Trainer
import trainer


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_loader, val_loader, test_loader, nclass = config.initialize('data_loader', module_data)

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch, nclass)
    # logger.debug(model)

    # get function handles of loss and metrics
    # loss = getattr(module_loss, config['loss'])

    if config['use_balanced_weights']:
        classes_weights_path = os.path.join(
            config['data_loader']['args']['data_dir'],
            config['data_loader']['args']['dataset'] + '_classes_weights.npy')
        if os.path.isfile(classes_weights_path):
            print(f'load weight from {classes_weights_path}')
            weight = np.load(classes_weights_path)
        else:
            weight = calculate_weigths_labels(config, train_loader, nclass)
        weight = torch.from_numpy(weight.astype(np.float32))
    else:
        weight = None
    loss = config.initialize('loss', module_loss, weight).build_loss(mode=config['loss']['mode'])

    evaluator = getattr(module_metric, config['evaluator'])(num_class=nclass)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    # Trainer = config.initialize('trainer_type', trainer)
    Trainer = getattr(trainer, config['trainer_type'])

    trainer_ob = Trainer(model, loss, evaluator, optimizer,
                         config=config,
                         data_loader=train_loader,
                         valid_data_loader=val_loader,
                         lr_scheduler=lr_scheduler)

    trainer_ob.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
