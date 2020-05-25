import copy
import time
import math
import logging
import os
import re
import wget
import zipfile
from functools import partial
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchtext.data as data
import torchtext.vocab as vocab
from torch.nn.utils import clip_grad_value_, clip_grad_norm_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataset(glove_embedding_name='6B', dim=300):
    """
        Downloads the dataset with 3000 reviews from IMDb, Yelp and Amazon
        and returnd a torchtext.data.Dataset object and word vectors.

        Args:
            glove_embedding_name - str; it tells the size of the dataset that
            the original word2vec model was trained on. Default is '6B'.

            dim - int; dimension of the word embedding

            ---> See the torchtext docs for a list of available pretrained
            embeddings.

        Returns:
            dataset - torchtext.data.Dataset object.

            word_vectors - torch.tensor; It contains only those vectors
            corresponding to words that appear in the dataset. Words that appear
            in the dataset but not in the GloVe vocabulary are replaced by
            the <unk> token, whose word vector is initialised from a normal
            distribution. The <pad> token is added and its word vector is
            initialised with zeros.
    """
    logging.info('Downloading data')
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.listdir('data'):
        wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment labelled sentences.zip', out='data/data.zip')
        with zipfile.ZipFile('data/data.zip', 'r') as myzip:
            myzip.extractall('data/')

    # separate lines into reviews and class labels
    path = os.path.abspath('data/sentiment labelled sentences')
    regex = re.compile(r'^(.*?)\s+(\d)$')
    reviews = []
    for file in ['imdb_labelled.txt', 'amazon_cells_labelled.txt', 'yelp_labelled.txt']:
        with open(os.path.join(path, file)) as f:
            for line in f:
                result = regex.match(line)
                reviews.append([result.group(1), int(result.group(2))])

    TEXT = data.Field(tokenize='spacy', lower=True, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    fields = [('review', TEXT), ('label', LABEL)]

    logging.info('Downloading GloVe word vectors')
    GLOVE = vocab.GloVe(name=glove_embedding_name, dim=dim)

    # pipeline for replacing words that do not appear in GloVe vocab
    def replace_unknown(x, word_list):
        if x in word_list:
            return x
        else:
            return '<unk>'

    pipe = data.Pipeline(partial(replace_unknown, word_list=list(GLOVE.stoi.keys())))
    TEXT.preprocessing = pipe

    examples = [data.Example.fromlist(review, fields) for review in reviews]
    dataset = data.Dataset(examples, fields)

    TEXT.build_vocab(dataset, vectors='glove.6B.{}d'.format(dim))
    TEXT.vocab.vectors[0] = torch.normal(mean=TEXT.vocab.vectors.mean()*torch.ones(1, dim),  # <unk> token
                                         std=TEXT.vocab.vectors.std()*torch.ones(1, dim))
    TEXT.vocab.vectors[1] = torch.zeros(1, dim)  # <pad> token

    return dataset, TEXT.vocab.vectors


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
        dl = MyIterator(ds, bs, sort_key=lambda x: len(x.review),
                        shuffle=False, train=False, device=device)
        for i, batch in enumerate(dl):
            output = model(batch.review)
            predictions.extend(output.argmax(dim=1).tolist())
            gt.extend(batch.label.tolist())
            loss += loss_fn(output, batch.label).item()
        avg_loss = loss/(i+1)

    accuracy = np.mean(np.array(predictions) == np.array(gt))
    if is_in_train:
        model.train()

    return avg_loss, accuracy


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

    def train(self, epochs=1, bs=8, grad_clip=None,
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
        for epoch in range(epochs):

            total_loss = 0
            dl_train = MyIterator(self.ds_train, bs, shuffle=True,
                                  sort_key=lambda x: len(x.review),
                                  device=self.device)
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
                        self.recorder.record(val_loss=val_loss,
                                             val_loss_batch=epoch*no_batches + i)
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
                  no_points='auto', beta=0.7):
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
            self.optim.zero_grad()

            output = self.model(batch.review)
            loss = self.loss_fn(output, batch.label)
            if i == 1:
                initial_loss = loss.item()
            ema = (beta * ema + (1 - beta) * loss.item())
            ema_smoothed = ema / (1 - beta**i)

            if i == no_points:
                break
            elif ema_smoothed > 1.1 * initial_loss:
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

    def plot_losses(self, show_lr=False, raw=True, exp_avg=True, beta=0.8):
        plt.figure(figsize=(8, 6))

        if raw:
            plt.plot(self['train_loss'], label='train loss')

        if exp_avg:
            ema = exp_mov_avg(self['train_loss'], beta)
            plt.plot(ema, 'r', label='avg train loss')

        if self['val_loss'] != []:
            plt.plot(self['val_loss_batch'], self['val_loss'],
                     'm-o', label='val loss')

        plt.grid(b=True, which='major')
        plt.legend()
        plt.show()

        if show_lr:
            plt.figure(figsize=(8, 6))
            plt.plot(self['lr'])
            plt.grid(b=True, which='major')
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


def get_fold_data(ds, n_folds, random_state=None):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    examples = np.array(ds.examples)
    fields = ds.fields
    for train_index, val_index in kf.split(examples):
        yield (data.Dataset(examples[train_index], fields=fields),
               data.Dataset(examples[val_index], fields=fields))


def cross_validate(dataset, model, loss_fn, training_fn,
                   n_folds, random_state=None):
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
                 .format(sum(losses['train'])/n_folds,
                         sum(accuracies['train'])/n_folds))
    logging.info('Validation loss: {:.4f}, accuracy : {:.3f}'
                 .format(sum(losses['val'])/n_folds,
                         sum(accuracies['val'])/n_folds))
