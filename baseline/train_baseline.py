import os
import sys
path = os.path.dirname(os.path.abspath(__file__)) + '/../'
os.chdir(path)
sys.path.append(path)

import logging
import argparse
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils import cross_validate, validate, get_dataset
from learner import Learner
from schedulers import CyclicLRDecay


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Baseline_model(nn.Module):
    """
        Basic baseline model based on a bag of embeddings.

        Args:
            vocab_size - int; size of the dataset vocabulary .
            embed_dim - int; size of the embedding vectors.
            embed_vecs - tensor; pretrained word embeddings (optional).
    """

    def __init__(self, embed_vecs):
        super().__init__()
        self.embed_dim = embed_vecs.shape[1]
        self.embedding = nn.EmbeddingBag.from_pretrained(embed_vecs)
        self.head = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim//2),
                                  nn.ReLU(),
                                  nn.Linear(self.embed_dim//2, 2))

    def forward(self, batch):
        batch, lengths = batch

        trimmed_reviews = [review[:length] for review, length in zip(batch, lengths)]
        batch_concat = torch.cat(trimmed_reviews, dim=0)
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths[:-1], dim=0)])

        out = self.embedding(batch_concat, offsets)  # (B x E)
        return self.head(out)


def training(model, loss_fn, ds_train, ds_val):
    optimiser = AdamW(model.parameters(), weight_decay=1e-2)
    steps_per_cycle = 2*(len(ds_train)//32 + 1)
    scheduler = CyclicLRDecay(optimiser, 1e-4, 1e-2, steps_per_cycle, 0.25, gamma_factor=0.65)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=10, bs=32, grad_clip=(0.2, 0.7))

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    learner.train(epochs=8, bs=32, grad_clip=(0.2, 0.7))


def main():
    # Setup logger
    logging.basicConfig(filename='log.txt',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s : %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info('Running file : {}'.format(__file__))

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--randomseed',
                        type=int,
                        default=None,
                        help='random seed')
    parser.add_argument('-cv', '--crossvalidate',
                        action='store_true',
                        help='cross-validation does not save the model at the end!')
    parser.add_argument('-k',
                        type=int,
                        default=10,
                        help='number of folds, default=10')
    args = parser.parse_args()

    # get dataset
    dataset, emb_weights = get_dataset()
    dataset.fields['review'].include_lengths = True
    # the above line adds a tensor with lengths of reviews in a batch,
    # so that we can pass batches to embeddingbag

    if args.randomseed is not None:
        random.seed(args.randomseed)

    # create model and loss function
    model = Baseline_model(emb_weights.clone()).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    if args.crossvalidate:  # cross validation routine
        cross_validate(dataset, model, loss_fn, training, args.k)
    else:  # single split routine
        ds_train, ds_val = dataset.split(split_ratio=[0.9, 0.1])
        logging.info('Initialising the model with the embedding layer frozen.')
        training(model, loss_fn, ds_train, ds_val)
        logging.info('--- Final statistics ---')
        logging.info('Training loss : {:.4f}, accuracy {:.4f}'
                     .format(*validate(ds_train, loss_fn, model)))
        logging.info('Validation loss : {:.4f}, accuracy {:.4f}'
                     .format(*validate(ds_val, loss_fn, model)))

        if not os.path.exists('models'):
            os.makedirs('models')
        logging.info('Model saved to: models/model_baseline.pt')
        torch.save(model.state_dict(), 'models/model_baseline.pt')


if __name__ == '__main__':
    main()
