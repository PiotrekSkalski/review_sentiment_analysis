import copy
import logging
import os
import wget
import tarfile
import pickle
from functools import partial
from sklearn.model_selection import KFold
import numpy as np
import torch
import torchtext.data as data
import torchtext.vocab as vocab


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataset(min_freq=2, test_set=False):
    """
        Downloads the IMDB dataset and returns a torchtext.data.Dataset object
        and word vectors. It contains 50,000 files so this function might take
        a while if it's run for the first time.

        Args:
            min_freq - int; only words with frequency above min_freq are
            included in the vocabulary.

            test_set - bool; Whether to return the test dataset; default=False.

        Returns:
            dataset - torchtext.data.Dataset object.

            (dataset_test - torchtext.data.Dataset object; optional argument,
            only if test_set=True)

            word_vectors - torch.tensor; pretrained GloVe word vectors aligned
            with the vocabulary; from 'glove.840B.300d'.
    """

    logging.info('----- IMDb dataset -----')
    if not os.path.exists('data'):
        logging.info('Downloading data')
        os.makedirs('data')
        wget.download(
            'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
            out='data/aclImdb_v1.tar.gz'
        )
        logging.info('Untar data')
        os.system('tar -C data/ -xzf {}'.format('data/aclImdb_v1.tar.gz'))

    # separate lines into reviews and class labels
    def process_set(set_path, pickle_name):
        reviews = []
        sets = {'pos/': 1, 'neg/': 0}
        for folder, label in sets.items():
            files = os.listdir(os.path.join(set_path, folder))
            for file_name in files:
                with open(os.path.join(set_path, folder, file_name)) as f:
                    reviews.append([f.read(), label])
        with open(os.path.join(set_path, pickle_name), 'wb') as f:
            pickle.dump(reviews, f)

    train_path = os.path.abspath('data/aclImdb/train/')
    pickle_name_train = 'pickled_list_train.pkl'
    if not os.path.exists(os.path.join(train_path, pickle_name_train)):
        logging.info('Processing the training set')
        process_set(train_path, pickle_name_train)

    test_path = os.path.abspath('data/aclImdb/test/')
    pickle_name_test = 'pickled_list_test.pkl'
    if test_set and not os.path.exists(os.path.join(test_path, pickle_name_test)):
        logging.info('Processing the test set')
        process_set(test_path, pickle_name_test)

    with open(os.path.join(train_path, pickle_name_train), 'rb') as f:
        reviews = pickle.load(f)
    if test_set:
        with open(os.path.join(test_path, pickle_name_test), 'rb') as f:
            reviews_test = pickle.load(f)

    # build dataset
    TEXT = data.Field(tokenize='spacy', lower=True, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    fields = [('review', TEXT), ('label', LABEL)]

    logging.info('Building training dataset')
    examples = [data.Example.fromlist(review, fields) for review in reviews]
    dataset = data.Dataset(examples, fields)

    logging.info('Building vocab')
    GLOVE = vocab.GloVe('840B', 300)
    unk_init = partial(torch.Tensor.normal_,
                       mean=GLOVE.vectors.mean(),
                       std=GLOVE.vectors.std())
    GLOVE.unk_init = unk_init
    TEXT.build_vocab(dataset, vectors=GLOVE, min_freq=min_freq)
    TEXT.vocab.vectors[1] = torch.zeros((1, 300))  # pad token

    if test_set:
        logging.info('Building test dataset')
        examples_test = [data.Example.fromlist(review, fields) for review in reviews_test]
        dataset_test = data.Dataset(examples_test, fields)
        return dataset, dataset_test, TEXT.vocab.vectors

    return dataset, TEXT.vocab.vectors


class MyIterator(data.Iterator):
    """
        Custom iterator that tries to make batches with examples of
        similar length. First, it randomly splits into chunks of
        50*batch_size examples. Then sorts within each chunk and
        builds batches from these sorted examples. It then shuffles
        the batches and yields them.

        Code adopted from
        http://nlp.seas.harvard.edu/2018/04/03/attention.html#iterators
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
