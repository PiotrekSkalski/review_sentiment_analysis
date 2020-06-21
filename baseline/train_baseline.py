import os
import sys
path = os.path.dirname(os.path.abspath(__file__ + '/../'))
os.chdir(path)
sys.path.append(path)

import logging
import argparse
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils import validate, get_dataset
from learner import Learner
from schedulers import CyclicLRDecay


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Baseline_model(nn.Module):
    """
        Basic baseline model based on a bag of embeddings.
    """

    def __init__(self, embed_vecs):
        """
            Args:
                embed_vecs - tensor; pretrained word embeddings.
        """
        super().__init__()
        self.embed_dim = embed_vecs.shape[1]
        self.embedding = nn.EmbeddingBag.from_pretrained(embed_vecs)
        self.head = nn.Linear(self.embed_dim, 2)

    def forward(self, batch):
        """
            Args:
                batch - tuple: (batch, lengths);
        """
        batch, lengths = batch

        trimmed_reviews = [review[:length] for review, length in zip(batch, lengths)]
        batch_concat = torch.cat(trimmed_reviews, dim=0)
        offsets = torch.cat([torch.tensor([0]).to(device), torch.cumsum(lengths[:-1], dim=0)])

        out = self.embedding(batch_concat, offsets)  # (B x E)
        return self.head(out)


def main():
    # Setup logger
    logging.basicConfig(filename='log.txt',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s : %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info('Running file : {}'.format(__file__))

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='random seed')
    parser.add_argument('--save-to',
                        type=str,C>
                        default=None,
                        help='name of a file to save')
    args = parser.parse_args()

    # get dataset
    dataset, test_dataset, emb_weights = get_dataset(min_freq=5, test_set=True)
    dataset.fields['review'].include_lengths = True
    # the above line adds a tensor with lengths of reviews in a batch,
    # so that we can pass batches to embeddingbag
    ds_train, ds_val = dataset.split(split_ratio=[0.9, 0.1])

    if args.seed is not None:
        random.seed(args.randomseed)

    bs = 64
    logging.info('Initialising the model with the embedding layer frozen.')
    model = Baseline_model(emb_weights.clone()).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimiser = AdamW(model.parameters(), weight_decay=1e-2)
    steps_per_cycle = (len(ds_train)//bs + 1)
    scheduler = CyclicLRDecay(optimiser, 1e-4, 1e-2, steps_per_cycle, 0.25, gamma_factor=0.80)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=6, bs=bs, grad_clip=(0.2, 0.4))

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    learner.train(epochs=5, bs=bs, grad_clip=(0.2, 0.4))

    logging.info('--- Final statistics ---')
    logging.info('Training loss : {:.4f}, accuracy {:.4f}'
                 .format(*validate(ds_train, loss_fn, model, bs)))
    logging.info('Test loss : {:.4f}, accuracy {:.4f}'
                 .format(*validate(test_dataset, loss_fn, model, bs)))

    if args.save_to is not None:
        if not os.path.exists('models'):
            os.makedirs('models')
        logging.info('Model saved to: models/{}'.format(args['--save-to']))
        torch.save(model.state_dict(), 'models/{}'.format(args['--save-to']))


if __name__ == '__main__':
    main()
