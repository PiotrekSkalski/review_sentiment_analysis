import os
import sys
path = os.path.dirname(os.path.abspath(__file__)) + '/../'
os.chdir(path)
sys.path.append(path)

import random
import logging
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils import validate, cross_validate, get_dataset
from learner import Learner
from schedulers import CyclicLRDecay


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, query_dim, n_outputs, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(query_dim, query_dim//2)
        self.W2 = nn.Linear(query_dim//2, n_outputs)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query):
        attn_weights = self.W2(self.dropout(torch.tanh(self.W1(query))))
        attn_weights = attn_weights.permute(0, 2, 1)

        return self.softmax(attn_weights)


class GRU_SelfAttention_model(nn.Module):
    def __init__(self, embed_vecs, hidden_size=512, num_layers=1, attn_output_size=8,
                 dropout=(0.2, 0.2), bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attn_output_size = attn_output_size
        self.embed_dim = embed_vecs.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embed_vecs)
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout[0],
                          bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(p=dropout[1])
        self.attention = SelfAttention(self.num_directions*self.hidden_size,
                                       self.attn_output_size, dropout[0])
        self.head = nn.Linear(self.attn_output_size*self.num_directions*self.hidden_size, 2)

    def forward(self, batch):
        batch, lengths = batch
        batch_dim, _ = batch.shape

        embedded = self.dropout(self.embedding(batch))  # (B x E)
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        outputs_packed, hiddens = self.gru(embedded_packed)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        # outputs : (B x L x H)
        # hiddens : (L x B x H)

        attn_weights = self.attention(self.dropout(outputs))  # (B x A x L)
        attn_output = torch.bmm(attn_weights, outputs).view(batch_dim, -1)  # (B x A x L) x (B x L x H) -> (B x A x H) -> (B x A*H)

        logging.debug('batch shape : {}'.format(batch.shape))
        logging.debug('embedding shape : {}'.format(embedded.shape))
        logging.debug('hiddens shape : {}'.format(hiddens.shape))
        logging.debug('outputs shape : {}'.format(outputs.shape))
        logging.debug('attn_weights shape : {}'.format(attn_weights.shape))
        logging.debug('attn_output shape : {}'.format(attn_output.shape))

        return self.head(self.dropout(attn_output))


def training(model, loss_fn, ds_train, ds_val):
    optimiser = AdamW(model.parameters(), weight_decay=1e-2)
    cycle_steps = 2*(len(ds_train)//32 + 1)
    scheduler = CyclicLRDecay(optimiser, 1e-4, 1e-2, cycle_steps, 0.25, gamma_factor=0.6)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=8, bs=32, grad_clip=(0.10, 1.2))

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    learner.train(epochs=4, bs=32, grad_clip=(0.10, 1.2))


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
    model = GRU_SelfAttention_model(emb_weights.clone(), bidirectional=True, num_layers=1,
                                    hidden_size=32, attn_output_size=8, dropout=(0, 0.5)).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # cross validation routine
    if args.crossvalidate:
        cross_validate(dataset, model, loss_fn, training, args.k)
    # single split routine
    else:
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
        logging.info('Model saved to: models/model_GRU_selfattention.pt')
        torch.save(model.state_dict(), 'models/model_GRU_selfattention.pt')


if __name__ == '__main__':
    main()
