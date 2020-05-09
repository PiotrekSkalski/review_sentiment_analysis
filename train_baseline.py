import os
import logging
import argparse
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import validate, learner, get_fold_data, cross_validate
from get_dataset import get_dataset


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
    optimiser = Adam(model.parameters(), lr=2e-4)
    logging.info('Training, lr=2e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=10, bs=8)

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    optimiser.param_groups[0]['lr'] = 1e-4
    logging.info('Training, lr=1e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=10, bs=8)


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

    # cross validation routine
    if args.crossvalidate:
        cross_validate(dataset, model, loss_fn, training, args.k, args.randomseed)
    # single split routine
    else:
        ds_train, ds_val = dataset.split(split_ratio=[0.9, 0.1],
                                         random_state=random.getstate())
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
