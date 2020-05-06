import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.data import BucketIterator
import torch
from torch.nn.utils import clip_grad_value_
import time
import logging



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(ds, loss_fn, model, bs=1, device=device):
    """
        Loops over a dataset (validation or test) and evaluates average
        loss and accuracy of a given model.
    """
    is_in_train = model.training
    model.eval()
    with torch.no_grad():
#         size = len(ds)
        predictions = []
        gt = list(ds.label)
        loss = 0
        for i, batch in enumerate(BucketIterator(ds, bs, sort_key=lambda x: len(x.review), shuffle=False, device=device)):
            output = model(batch.review)
            predictions.extend(output.argmax(dim=1).tolist())
            loss += loss_fn(output, batch.label).item()
        avg_loss = loss/(i+1)

    accuracy = np.mean(np.array(predictions) == np.array(gt))
    if is_in_train: model.train()
        
    return avg_loss, accuracy


def lr_finder(model, dataset, optimiser, loss_fn, lr_range=[1e-6, 1e0], bs=1, avg_over_batches=200, device=device):
    """
        A utility that helps find a good learning rate. It gradually increases the learning rate
        every 'avg_over_batches' batches and records how the loss evolves. Usually one sees a
        decrease in loss at some point, plateauing, and then increase once the lr is too big.
    """
    lr_list = np.logspace(np.log10(lr_range[0]),
                          np.log10(lr_range[1]),
                         np.int(np.log10(lr_range[1]/lr_range[0])*10))
    initial_state_dict = copy.deepcopy(model.state_dict())
    losses = []
    learning_rates = []
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr_list[0]
    
    tot_batches_todo = len(lr_list)*avg_over_batches
    for i, batch in enumerate(BucketIterator(dataset, bs, sort_key=lambda x: len(x.review), shuffle=True, device=device, repeat=True), 1):
        optimiser.zero_grad()

        output = model(batch.review)
        loss = loss_fn(output, batch.label)
        losses.append(loss.item())
        learning_rates.append(optimiser.param_groups[0]['lr'])

        loss.backward()
        optimiser.step()
        
        print('Finding learning rate ...{:.0f}%\r'.format(i/tot_batches_todo*100), end='')

        if i == 1:
            initial_loss = loss.item()
        elif loss.item() > 100*initial_loss:
            break
        elif (i // avg_over_batches) == len(lr_list):
            break
        elif not i % avg_over_batches:
            for param_group in optimiser.param_groups:
                param_group['lr'] = lr_list[i // avg_over_batches]
  
    model.load_state_dict(initial_state_dict)
    sns.lineplot(learning_rates, losses)
    plt.xscale('log')

def learner(model, loss_fn, optimiser, ds_train, ds_val=None, epochs=1, bs=4, device=device, grad_clip=None):
    """
        A basic training loop that logs some training statistics, like losses and accuracy.
        
        Args:
            model - a model class that takes one argument to its forwards method.
            loss_fn - torch.nn module.
            optimiser - torch.optim module.
            ds_train - torchtext.data.Dataset used for training.
            ds_val - torchtext.data.Dataset used for validation, default=None.
            epochs - int; default=1.
            bs - int; batch_size, default=4.
            device - torch.device;
            grad_clip - float; maximum value for a gradient clipping method,
                        default=none, i.e. no grad clipping.
                        
    """
    start_time = time.time()
    for epoch in range(epochs):
        
        total_loss = 0
        for i, batch in enumerate(BucketIterator(ds_train, bs, sort_key=lambda x: len(x.review),
                                                 shuffle=True, device=device), 1):
            optimiser.zero_grad()
            
            output = model(batch.review)
            loss = loss_fn(output, batch.label)
            total_loss += loss.item()
        
            loss.backward()
            if grad_clip is not None:
                clip_grad_value_(model.parameters(), grad_clip)
            optimiser.step()
            
            # Logs statistics 3 times during an epoch
            if not i % (len(ds_train)//(bs*3)):
                avg_loss = total_loss / (len(ds_train)//(bs*3))
                total_loss = 0
                if ds_val is not None:
                    val_loss, val_accuracy = validate(ds_val, loss_fn, model, bs=bs, device=device)
                    logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, val_loss = {:.4f}, '
                                 'val_accuracy : {:.3f}, time = {:.0f}s'
                                 .format(epoch + 1, i, avg_loss, val_loss, val_accuracy, time.time() - start_time))
                else:
                    logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, time = {:.0f}s'
                                 .format(epoch + 1, i, avg_loss, time.time() - start_time))