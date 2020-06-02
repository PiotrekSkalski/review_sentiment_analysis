import copy
import time
import math
import logging
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from utils import MyIterator, validate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Learner:
    """
    """
    def __init__(self, model, loss_fn, optim, lr_sched,
                 ds_train, ds_val=None, device=device):
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr_sched = lr_sched
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.device = device

        self.recorder = Recorder()

    def train(self, epochs=1, bs=8, lr=None, grad_clip=None,
              silent=False, no_logs_per_epoch=4):
        """
            A basic training loop that logs some training statistics,
            like losses and accuracy.

            Args:
                epochs - int; default=1.
                bs - int; batch_size, default=4.
                grad_clip - float; maximum value for a gradient clipping method,
                            default=none, i.e. no grad clipping.

        """
        start_time = time.time()
        no_batches = math.ceil(len(self.ds_train)/bs)
        if lr is not None:
            self.optim.param_groups[0]['lr'] = lr

        dl_train = MyIterator(self.ds_train, bs, shuffle=True,
                              sort_key=lambda x: len(x.review),
                              device=self.device)

        for epoch in range(epochs):

            total_loss = 0
            for i, batch in enumerate(dl_train, 1):
                self.optim.zero_grad()

                output = self.model(batch.review)
                loss = self.loss_fn(output, batch.label)
                total_loss += loss.item()

                loss.backward()

                if grad_clip is not None:
                    if grad_clip[0] is not None:
                        clip_grad_value_(self.model.parameters(), grad_clip[0])
                    if grad_clip[1] is not None:
                        clip_grad_norm_(self.model.parameters(), grad_clip[1])

                grads = [param.grad for param in self.model.parameters()]
                grads = torch.cat([grad.flatten() for grad in grads
                                   if grad is not None])
                self.recorder.record(train_loss=loss.item(),
                                     lr=self.optim.param_groups[0]['lr'],
                                     grad_max=grads.max().item(),
                                     grad_norm=grads.norm(p=2).item())

                self.optim.step()
                if self.lr_sched is not None:
                    self.lr_sched.step()

                # Logs statistics
                if not i % (no_batches//no_logs_per_epoch):
                    avg_loss = total_loss / (no_batches//no_logs_per_epoch)
                    total_loss = 0
                    if self.ds_val is not None:
                        val_loss, val_accuracy = validate(self.ds_val, self.loss_fn,
                                                          self.model, bs=bs,
                                                          device=device)
                        if self.recorder['val_loss_batch'] == []:
                            prev_batch_no = 0
                        else:
                            prev_batch_no = self.recorder['val_loss_batch'][-1]
                        self.recorder.record(
                            val_loss=val_loss,
                            val_loss_batch=prev_batch_no + no_batches//no_logs_per_epoch
                        )
                        if not silent:
                            logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, val_loss = {:.4f}, '
                                         'val_accuracy : {:.3f}, time = {:.0f}s'
                                         .format(epoch + 1, i, avg_loss, val_loss,
                                                 val_accuracy, time.time() - start_time))
                    else:
                        if not silent:
                            logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, time = {:.0f}s'
                                         .format(epoch + 1, i, avg_loss, time.time() - start_time))

    def lr_finder(self, bs=32, lr_range=(1e-6, 1e0),
                  no_points='auto', beta=0.7, early_stop=True):
        initial_state_dict = copy.deepcopy(self.model.state_dict())
        initial_lr = self.optim.param_groups[0]['lr']
        self.optim.param_groups[0]['lr'] = lr_range[0]

        if no_points == 'auto':
            no_points = np.int(np.log10(lr_range[1]/lr_range[0])*100)
        mult_factor = np.power(lr_range[1]/lr_range[0], 1/no_points)

        dl = MyIterator(self.ds_train, bs, sort_key=lambda x: len(x.review),
                        shuffle=True, device=device, repeat=True)

        self.recorder.update({'lr_finder_loss': [],
                              'lr_finder_lr': []})
        ema = 0
        for i, batch in enumerate(dl, 1):
            print('{:.2f}%\r'.format(i/no_points), end='')
            self.optim.zero_grad()

            output = self.model(batch.review)
            loss = self.loss_fn(output, batch.label)
            if i == 1:
                initial_loss = loss.item()
            ema = (beta * ema + (1 - beta) * loss.item())
            ema_smoothed = ema / (1 - beta**i)

            if i == no_points:
                break
            elif early_stop and ema_smoothed > 1.1 * initial_loss:
                break

            loss.backward()

            self.recorder.record(lr_finder_loss=ema_smoothed,
                                 lr_finder_lr=self.optim.param_groups[0]['lr'])

            self.optim.step()
            self.optim.param_groups[0]['lr'] *= mult_factor

        self.model.load_state_dict(initial_state_dict)
        self.optim.param_groups[0]['lr'] = initial_lr

        plt.plot(self.recorder['lr_finder_lr'],
                 self.recorder['lr_finder_loss'])
        plt.xscale('log')
        plt.show()


class Recorder(dict):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(train_loss=[], val_loss=[], val_loss_batch=[],
                         lr=[], grad_max=[], grad_norm=[], **kwargs)

    def record(self, **kwargs):
        for key, value in kwargs.items():
            self[key].append(value)

    def plot_losses(self, show_lr=False, raw=True, gauss_avg=True, sigma=20):
        plt.figure(figsize=(8, 6))

        if raw:
            plt.plot(self['train_loss'], label='train loss')

        if gauss_avg:
            smoothed = gaussian_filter1d(self['train_loss'], sigma)
            plt.plot(smoothed, 'tab:orange', linewidth=3, label='avg train loss')

        if self['val_loss'] != []:
            plt.plot(self['val_loss_batch'], self['val_loss'],
                     'r', linewidth=3, label='val loss')

        plt.grid(b=True, which='major')
        plt.legend()
        plt.show()

        if show_lr:
            plt.figure(figsize=(8, 6))
            plt.plot(self['lr'])
            plt.grid(b=True, which='major')
            plt.show()

    def plot_grads(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self['grad_max'])
        plt.ylabel('max gradient')
        plt.show()
        plt.figure(figsize=(8, 6))
        plt.plot(self['grad_norm'])
        plt.ylabel('norm of gradient')
        plt.show()

    def reset(self):
        self.__init__()


def exp_mov_avg(values, beta):
    ema = 0
    ema_unbiased = []
    for i, value in enumerate(values, 1):
        ema = beta * ema + (1 - beta) * value
        ema_unbiased.append(ema / (1 - beta**i))
    return ema_unbiased
