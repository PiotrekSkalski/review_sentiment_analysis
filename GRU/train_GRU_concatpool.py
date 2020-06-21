import os
import sys
path = os.path.dirname(os.path.abspath(__file__ + '/../'))
os.chdir(path)
sys.path.append(path)

import logging
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils import validate, get_dataset
from learner import Learner
from schedulers import CyclicLRDecay


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRUConcatPool(nn.Module):
    """
        A bidirectional GRU with single layer and a concalpool layer.
        The head is a FCNN with two linear layers.
    """
    def __init__(self, embed_vecs=None, hidden_size=512, dropout=(0, 0, 0)):
        """
            Args:
                embed_vecs - tensor; pretrained embedding vectors.
                hidden_size - int; hidden size of the GRU module.
                dropout - tuple(float, float, float); dropout applied after
                the embedding layer, GRU layer and between the linear layers
                in the head.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(embed_vecs)

        self.gru = nn.GRU(input_size=embed_vecs.size()[1], hidden_size=hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True)
        self.emb_dropout = nn.Dropout(p=dropout[0])
        self.dropout = nn.Dropout(p=dropout[1])
        self.head = nn.Sequential(
            nn.Linear(4*self.hidden_size, 2*self.hidden_size),
            nn.Dropout(p=dropout[2]),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, 2)
        )

    def forward(self, batch):
        """
            Args:
                batch - tuple(batch, lengths);
        """
        batch, lengths = batch
        batch_dim, _ = batch.shape

        embedded = self.emb_dropout(self.embedding(batch))
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        outputs_packed, hiddens = self.gru(embedded_packed)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        outputs = self.dropout(outputs) # B x L x 2H

        avg_pool = torch.sum(outputs, dim=1)/lengths.unsqueeze(1).to(device) # B x 2H

        # masking
        mask = torch.zeros_like(outputs, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, length:, :] = True
        outputs.masked_fill_(mask, -float('inf')) # B x L x 2H
        max_pool = torch.max(outputs, dim=1)[0] # B x 2H

        cat_output = torch.cat([avg_pool, max_pool], dim=1) # B x 6H

        logging.debug('batch shape : {}'.format(batch.shape))
        logging.debug('embedding shape : {}'.format(embedded.shape))
        logging.debug('hiddens shape : {}'.format(hiddens.shape))
        logging.debug('outputs shape : {}'.format(outputs.shape))
        logging.debug('avg_pool shape : {}'.format(avg_pool.shape))
        logging.debug('max_pool shape : {}'.format(max_pool.shape))

        return self.head(cat_output)


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
    model = GRUConcatPool(emb_weights.clone(), hidden_size=512,
                          dropout=(0.65, 0.65, 0.65)).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimiser = AdamW(model.parameters(), weight_decay=1e-3)
    cycle_steps = (len(ds_train)//bs + 1)
    scheduler = CyclicLRDecay(optimiser, 5e-5, 4e-3, cycle_steps, 0.1, gamma_factor=0.75)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=2, bs=bs, grad_clip=(0.4, 5))

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    learner.train(epochs=6, bs=bs, grad_clip=(0.4, 5))

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
