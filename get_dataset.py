import os
import re
import wget
import zipfile
from functools import partial
import torch
import torchtext.data as data
import torchtext.vocab as vocab
import logging


def replace_unknown(x, word_list):
    if x in word_list:
        return x
    else:
        return '<unk>'


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
    pipe = data.Pipeline(partial(replace_unknown, word_list=list(GLOVE.stoi.keys())))
    TEXT.preprocessing = pipe

    examples = [data.Example.fromlist(review, fields) for review in reviews]
    dataset = data.Dataset(examples, fields)

    TEXT.build_vocab(dataset, vectors='glove.6B.{}d'.format(dim))
    TEXT.vocab.vectors[0] = torch.normal(mean=TEXT.vocab.vectors.mean()*torch.ones(1, dim),  # <unk> token
                                         std=TEXT.vocab.vectors.std()*torch.ones(1, dim))
    TEXT.vocab.vectors[1] = torch.zeros(1, dim)  # <pad> token

    return dataset, TEXT.vocab.vectors
