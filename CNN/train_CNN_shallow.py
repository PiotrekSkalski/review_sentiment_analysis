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


class CNN(nn.Module):
    """
        A shallow CNN network.
    """
    def __init__(self, emb_weights, out_channels=100, kernel_heights=(3, 5, 7), dropout=(0, 0)):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(emb_weights)
        convolutions = []
        emb_dim = emb_weights.shape[1]
        for kernel in kernel_heights:
            padding = (kernel - 1) // 2
            convolutions.append(nn.Conv1d(emb_dim, out_channels, kernel, padding=padding))
        self.convolutions = nn.ModuleList(convolutions)
        self.emb_dropout = nn.Dropout(p=dropout[0])
        self.relu = nn.ReLU()
        self.head = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, len(kernel_heights)*out_channels//2),
            nn.Dropout(p=dropout[1]),
            nn.ReLU(),
            nn.Linear(len(kernel_heights)*out_channels//2, 2)
            )

    def forward(self, batch):
        batch, lengths = batch

        embedded = self.emb_dropout(self.embedding(batch)) # B x L x E
        embedded = embedded.transpose(1, 2) # B x E x L
        max_out = [self.conv_block(embedded, self.convolutions[i]) for i in range(len(self.convolutions))]
        cat_out = torch.cat(max_out, dim=1) # B x C*K

        return self.head(cat_out)

    def conv_block(self, input, conv_layer):
        # think about adding masking before max
        output = conv_layer(input)
        output = self.relu(output)
        output = torch.max(output, dim=2)[0]
        return output


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
    model = CNN(emb_weights.clone(), out_channels=128,
                kernel_heights=(1, 3, 5), dropout=(0.55, 0.5)).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimiser = AdamW(model.parameters(), weight_decay=1e-3)
    cycle_steps = (len(ds_train)//bs + 1)
    scheduler = CyclicLRDecay(optimiser, 6e-5, 3e-3, cycle_steps, 0.1, gamma_factor=0.7)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=2, bs=bs, grad_clip=(0.5, 2))

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    learner.train(epochs=8, bs=bs, grad_clip=(0.5, 2))

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
