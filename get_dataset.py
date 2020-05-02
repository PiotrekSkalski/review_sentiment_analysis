import os
import re
import wget
import zipfile
from functools import partial
import torch
import torchtext.data as data
import torchtext.vocab as vocab


def find_unknown(x, word_list):
  if x in word_list:
    return x
  else:
    return '<unk>'


def get_dataset():
  if not os.path.exists('data'):
    os.makedirs('data')
  if not os.listdir('data'):
    zip_file = wget.download('https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment labelled sentences.zip', out='data/data.zip')
    with zipfile.ZipFile('data/data.zip', 'r') as myzip:
      myzip.extractall('data/')
  
  path = os.path.abspath('data/sentiment labelled sentences')
  regex = re.compile(r'^(.*?)\s+(\d)$')
  reviews = []
  for file in ['imdb_labelled.txt', 'amazon_cells_labelled.txt', 'yelp_labelled.txt']:
    with open(os.path.join(path, file)) as f:
      for line in f:
        result = regex.match(line)
        reviews.append([result.group(1), int(result.group(2))])

  TEXT = data.Field(tokenize='spacy', lower=True, batch_first=True)
  TEXT.vocab = vocab.GloVe(name='6B', dim=300)
  TEXT.vocab.stoi[TEXT.unk_token] = len(TEXT.vocab.itos)
  TEXT.vocab.itos.append(TEXT.unk_token)
  TEXT.vocab.stoi[TEXT.pad_token] = len(TEXT.vocab.itos)
  TEXT.vocab.itos.append(TEXT.pad_token)
  init_vectors = torch.normal(mean=TEXT.vocab.vectors.mean()*torch.ones(2,300),
                              std=TEXT.vocab.vectors.std()*torch.ones(2,300))
  TEXT.vocab.vectors = torch.cat([TEXT.vocab.vectors, init_vectors], dim=0)
  LABEL = data.Field(sequential=False, use_vocab=False)
  fields = [('review', TEXT), ('label', LABEL)]

  pipe = data.Pipeline(partial(find_unknown, word_list=TEXT.vocab.stoi.keys()))
  TEXT.preprocessing = pipe

  examples = [data.Example.fromlist(review, fields) for review in reviews]
  dataset = data.Dataset(examples, fields)

  return dataset, TEXT.vocab.vectors
