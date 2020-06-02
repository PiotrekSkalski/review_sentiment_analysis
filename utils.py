import copy
import logging
import os
import re
import wget
import zipfile
from functools import partial
from sklearn.model_selection import KFold
import numpy as np
import torch
import torchtext.data as data
import torchtext.vocab as vocab


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
                        train=False, device=device)
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


def get_fold_data(ds, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=True)
    examples = np.array(ds.examples)
    fields = ds.fields
    for train_index, val_index in kf.split(examples):
        yield (data.Dataset(examples[train_index], fields=fields),
               data.Dataset(examples[val_index], fields=fields))


def cross_validate(dataset, model, loss_fn, training_fn, n_folds):
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    iterator = get_fold_data(dataset, n_folds)

    for i, (ds_train, ds_val) in enumerate(iterator, 1):
        logging.info('Cross-validating, fold : {}/{}'.format(i, n_folds))
        logging.info('Initialising the model with the embedding layer frozen.')
        fold_model = copy.deepcopy(model)
        training_fn(fold_model, loss_fn, ds_train, ds_val)
        
        for ds, ds_id in zip([ds_train, ds_val], ['train', 'val']):
            loss, acc = validate(ds, loss_fn, fold_model)
            losses[ds_id].append(loss)
            accuracies[ds_id].append(acc)
            
    logging.info('--- Cross-validation statistics ---')
    logging.info('Training loss: {:.4f}, accuracy : {:.3f}'
                 .format(sum(losses['train'])/n_folds,
                         sum(accuracies['train'])/n_folds))
    logging.info('Validation loss: {:.4f}, accuracy : {:.3f}'
                 .format(sum(losses['val'])/n_folds,
                         sum(accuracies['val'])/n_folds))
