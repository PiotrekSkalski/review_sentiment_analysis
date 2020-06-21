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


class ResBlock(nn.Module):
    """
        A residual block with skip connections.
    """
  def __init__(self, in_channels, out_channels, kernel=3,
               pool=False, skip=True, dropout=0):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel = kernel

    if pool:
      self.stride = 2
      self.pool = nn.Sequential(
          nn.MaxPool1d(3, stride=self.stride, padding=1),
          nn.Conv1d(in_channels, out_channels, 1)
      )
    else:
      self.stride = 1
      self.pool = None

    if skip:
      self.skip = nn.Conv1d(in_channels, out_channels,
                            kernel_size=1, stride=self.stride)
    else:
      self.skip = None

    self.conv_block = nn.Sequential(
        nn.Conv1d(in_channels, in_channels, kernel, padding=(kernel - 1)//2),
        nn.BatchNorm1d(in_channels),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Conv1d(in_channels, in_channels, kernel, padding=(kernel - 1)//2),
        nn.BatchNorm1d(in_channels),
        nn.Dropout(dropout),
        nn.ReLU()
    )

  def forward(self, input):
    out = self.conv_block(input)

    if self.pool is not None:
      out = self.pool(out)
    if self.skip is not None:
      out += self.skip(input)

    return out


class DCNN(nn.Module):
    """
        A deep CNN network made of residual blocks.
    """N?
  def __init__(self, emb_weights, in_channels, dropout=(0, 0, 0)):
    super().__init__()

    self.embedding = nn.Embedding.from_pretrained(emb_weights)
    self.emb_dim = emb_weights.shape[1]
    self.emb_dropout = nn.Dropout(p=dropout[0])

    self.stack = nn.Sequential(
        nn.Conv1d(self.emb_dim, in_channels, kernel_size=3, padding=1),
        ResBlock(in_channels, in_channels, dropout=dropout[1]),
        ResBlock(in_channels, 2*in_channels, pool=True, dropout=dropout[1]),
        ResBlock(2*in_channels, 2*in_channels, dropout=dropout[1]),
        ResBlock(2*in_channels, 4*in_channels, pool=True, dropout=dropout[1]),
        ResBlock(4*in_channels, 4*in_channels, dropout=dropout[1]),
        ResBlock(4*in_channels, 8*in_channels, pool=True, dropout=dropout[1]),
        ResBlock(8*in_channels, 8*in_channels, dropout=dropout[1]),
        ResBlock(8*in_channels, 8*in_channels, dropout=dropout[1])
    )

    self.head = nn.Sequential(
        nn.Linear(8*in_channels, 4*in_channels),
        nn.Dropout(dropout[2]),
        nn.ReLU(),
        nn.Linear(4*in_channels, 2)
    )

  def forward(self, batch):
    batch, lengths = batch

    embedded = self.emb_dropout(self.embedding(batch)) # B x L x E
    embedded = embedded.transpose(1, 2) # B x E x L

    out = self.stack(embedded) # B x 4C x L//4
    out = torch.max(out, dim=2)[0]

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
                        type=str,
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
    model = DCNN(emb_weights.clone(), 64, (0.4, 0.1, 0.4)).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimiser = AdamW(model.parameters(), weight_decay=1e-3)
    cycle_steps = (len(ds_train)//bs + 1)
    scheduler = CyclicLRDecay(optimiser, 5e-5, 1e-3, cycle_steps, 0.1, gamma_factor=0.75)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=2, bs=bs, grad_clip=(0.7, 4))

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    learner.train(epochs=8, bs=bs, grad_clip=(0.7, 4))

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
