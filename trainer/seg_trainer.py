import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
# from model.metric import Evaluator
from tqdm import tqdm
from utils import inf_loop
from data_loader.utils import decode_seg_map_sequence
from model.metric import SegEvaluator


class SegTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, evaluator: SegEvaluator, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, evaluator, optimizer, config)
        self.config = config
        self.dataset = config['data_loader']['args']['dataset']
        # self.ncalss = nclass
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(len(data_loader)) * 2)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        if self.lr_scheduler is not None:
            print(f'lr: {self.lr_scheduler.get_lr()}')
        self.model.train()
        self.evaluator.reset()
        total_loss = 0
        total_metrics = np.zeros(len(self.evaluator))
        tbar = tqdm(self.data_loader, ascii=True)
        for batch_idx, sample in enumerate(tbar):
            data, target = sample['image'], sample['label']
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            tbar.set_description('Train loss: %.3f' % (total_loss / (batch_idx + 1)))

            output = output.data.cpu().numpy()
            target = target.cpu().numpy()
            output = np.argmax(output, axis=1)

            self.evaluator.add_batch(target, output)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu()[:4], nrow=2, normalize=True))
                grid = make_grid(decode_seg_map_sequence(output[:4], dataset=self.dataset), nrow=2, normalize=False)
                self.writer.add_image('pred', grid)
                grid = make_grid(decode_seg_map_sequence(target[:4], dataset=self.dataset), nrow=2, normalize=False)
                self.writer.add_image('label', grid)

            if batch_idx == self.len_epoch:
                break

        if epoch == 0:
            self.data_loader.dataset.set_use_cache(True)
        
        print('[Epoch: %d, numImages: %5d]' % (epoch, batch_idx * self.data_loader.batch_size + data.data.shape[0]))
        print('Loss: %.5f' % total_loss)
        self.writer.add_scalar_with_tag('loss_epoch/train', total_loss, epoch)
        for i, metric in enumerate(self.evaluator):
            mtr_val = metric()
            total_metrics[i] = mtr_val
            self.writer.add_scalar_with_tag(f'train/{metric.__name__}', mtr_val, epoch)

        log = {
            'loss': total_loss / self.len_epoch,
            # 'metrics': (total_metrics / self.len_epoch).tolist()
            'metrics': total_metrics.tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        log_step = int(np.sqrt(len(self.valid_data_loader)) * 2)
        self.model.eval()
        self.evaluator.reset()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.evaluator))
        with torch.no_grad():
            tbar = tqdm(self.valid_data_loader, desc='\r', ascii=True)
            for batch_idx, sample in enumerate(tbar):
                data, target = sample['image'], sample['label']
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()

                tbar.set_description('Test  loss: %.3f' % (total_val_loss / (batch_idx + 1)))

                output = output.data.cpu().numpy()
                target = target.cpu().numpy()
                output = np.argmax(output, axis=1)

                self.evaluator.add_batch(target, output)
                if batch_idx % log_step == 0:
                    # total_val_metrics += self._eval_metrics(output, target)
                    self.writer.add_image('input', make_grid(data.cpu()[:4], nrow=2, normalize=True))
                    grid = make_grid(decode_seg_map_sequence(output[:4], dataset=self.dataset), nrow=2, normalize=False)
                    self.writer.add_image('pred', grid)
                    grid = make_grid(decode_seg_map_sequence(target[:4], dataset=self.dataset), nrow=2, normalize=False)
                    self.writer.add_image('label', grid)

        print('[Epoch: %d, numImages: %5d]' % (epoch, batch_idx * self.valid_data_loader.batch_size + data.data.shape[0]))
        print('Loss: %.5f' % total_val_loss)
        self.writer.add_scalar_with_tag('loss_epoch/val', total_val_loss, epoch)

        if epoch == 0:
            self.valid_data_loader.dataset.set_use_cache(True)

        for i, metric in enumerate(self.evaluator):
            mtr_val = metric()
            total_val_metrics[i] = mtr_val
            self.writer.add_scalar_with_tag(f'val/{metric.__name__}', mtr_val, epoch)
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        print(self.evaluator.confusion_matrix)
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            # 'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
            'val_metrics': total_val_metrics.tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
