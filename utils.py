import copy
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchtext.data as data
from torch.nn.utils import clip_grad_value_
import time
import logging


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyIterator(data.Iterator):
    """
        Custom iterator that tries to make batches with examples of
        similar length. First, it randomly splits into chunks of
        50*batch_size examples. Then sorts within each chunk and
        builds batches from these sorted examples. It then shuffles
        the batches and yields them.
    """
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 50):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def validate(ds, loss_fn, model, bs=1, device=device):
    """
        Loops over a dataset (validation or test) and evaluates average
        loss and accuracy of a given model.
    """
    is_in_train = model.training
    model.eval()
    with torch.no_grad():
        predictions = []
        gt = []
        loss = 0
        for i, batch in enumerate(MyIterator(ds, bs, sort_key=lambda x: len(x.review),
                                             shuffle=False, train=False, device=device)):
            output = model(batch.review)
            predictions.extend(output.argmax(dim=1).tolist())
            gt.extend(batch.label.tolist())
            loss += loss_fn(output, batch.label).item()
        avg_loss = loss/(i+1)

    accuracy = np.mean(np.array(predictions) == np.array(gt))
    if is_in_train:
        model.train()

    return avg_loss, accuracy


def lr_finder(model, dataset, optimiser, loss_fn, lr_range=[1e-6, 1e0],
              bs=1, avg_over_batches=200, device=device):
    """
        A utility that helps find a good learning rate. It gradually
        increases the learning rate every 'avg_over_batches' batches
        and records how the loss evolves. Usually one sees a decrease
        in loss at some point, plateauing, and then increase once the
        lr is too big.
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
    for i, batch in enumerate(MyIterator(dataset, bs, sort_key=lambda x: len(x.review),
                                         shuffle=True, device=device, repeat=True), 1):
        optimiser.zero_grad()

        output = model(batch.review)
        loss = loss_fn(output, batch.label)
        losses.append(loss.item())
        learning_rates.append(optimiser.param_groups[0]['lr'])

        loss.backward()
        optimiser.step()

        print('Finding learning rate ...{:.0f}%\r'
                     .format(i/tot_batches_todo*100), end='')

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


def learner(model, loss_fn, optimiser, ds_train, ds_val=None, epochs=1,
            bs=4, scheduler_fn=None, device=device, grad_clip=None):
    """
        A basic training loop that logs some training statistics,
        like losses and accuracy.

        Args:
            model - a model class that takes one argument to its forwards method.
            loss_fn - torch.nn module.
            optimiser - torch.optim module.
            ds_train - torchtext.data.Dataset used for training.
            ds_val - torchtext.data.Dataset used for validation, default=None.
            epochs - int; default=1.
            bs - int; batch_size, default=4.
            scheduler_fn - a function that takes optimiser as an argument and
                        returns a scheduler. It is run at the beginnign of
                        each epoch (use partial with standard schedulers)
            device - torch.device;
            grad_clip - float; maximum value for a gradient clipping method,
                        default=none, i.e. no grad clipping.

    """
    start_time = time.time()
    for epoch in range(epochs):

        if scheduler_fn is not None:
            scheduler = scheduler_fn(optimiser)
        total_loss = 0
        for i, batch in enumerate(MyIterator(ds_train, bs, sort_key=lambda x: len(x.review),
                                             shuffle=True, device=device), 1):
            optimiser.zero_grad()

            output = model(batch.review)
            loss = loss_fn(output, batch.label)
            total_loss += loss.item()

            loss.backward()
            if grad_clip is not None:
                clip_grad_value_(model.parameters(), grad_clip)
            optimiser.step()
            if scheduler_fn is not None:
                scheduler.step()

            # Logs statistics 3 times during an epoch
            if not i % (len(ds_train)//(bs*3)):
                avg_loss = total_loss / (len(ds_train)//(bs*3))
                total_loss = 0
                if ds_val is not None:
                    val_loss, val_accuracy = validate(ds_val, loss_fn, model, bs=bs, device=device)
                    logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, val_loss = {:.4f}, '
                                 'val_accuracy : {:.3f}, time = {:.0f}s'
                                 .format(epoch + 1, i, avg_loss, val_loss,
                                         val_accuracy, time.time() - start_time))
                else:
                    logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, time = {:.0f}s'
                                 .format(epoch + 1, i, avg_loss, time.time() - start_time))


def get_fold_data(ds, n_folds, random_state=None):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    examples = np.array(ds.examples)
    fields = ds.fields
    for train_index, val_index in kf.split(examples):
        yield (data.Dataset(examples[train_index], fields=fields),
               data.Dataset(examples[val_index], fields=fields))


def cross_validate(dataset, model, loss_fn, training_fn, n_folds, random_state=None):
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    iterator = get_fold_data(dataset, n_folds, random_state)
    original_model = model
    for i, (ds_train, ds_val) in enumerate(iterator, 1):
        logging.info('Cross-validating, fold : {}/{}'.format(i, n_folds))
        logging.info('Initialising the model with the embedding layer frozen.')
        model = copy.deepcopy(original_model)
        training_fn(model, loss_fn, ds_train, ds_val)
        for ds, ds_id in zip([ds_train, ds_val], ['train', 'val']):
            loss, acc = validate(ds, loss_fn, model)
            losses[ds_id].append(loss)
            accuracies[ds_id].append(acc)
    logging.info('--- Cross-validation statistics ---')
    logging.info('Training loss: {:.4f}, accuracy : {:.3f}'
                 .format(sum(losses['train'])/n_folds, sum(accuracies['train'])/n_folds))
    logging.info('Validation loss: {:.4f}, accuracy : {:.3f}'
                 .format(sum(losses['val'])/n_folds, sum(accuracies['val'])/n_folds))
