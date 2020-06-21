import os
import sys
path = os.path.dirname(os.path.abspath(__file__ + '/../'))
os.chdir(path)
sys.path.append(path)

import random
import logging
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils import validate, get_dataset
from learner import Learner
from schedulers import CyclicLRDecay


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    """
        A self attention module.
    """
    def __init__(self, query_dim, n_outputs):
        """
            Args:
                query_dim - int; dimension of the hidden layer of GRU module
                n_output - int; number of attention outputs.
        """
        super().__init__()
        self.W1 = nn.Linear(query_dim, query_dim//2, bias=False)
        self.W2 = nn.Linear(query_dim//2, n_outputs, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, lengths):
        attn_weights = self.W2(torch.tanh(self.W1(query)))
        attn_weights = attn_weights.permute(0, 2, 1)

        # masking the padding tokens
        mask = torch.zeros_like(attn_weights, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :, length:] = True
        attn_weights.masked_fill_(mask, -float('inf'))

        return self.softmax(attn_weights)


class GRUSelfAttention(nn.Module):
    """
        A bidirectional GRU with self attention on top and a FCNN head
        with two Linear layers.
    """
    def __init__(self, embed_vecs=None, hidden_size=256, attn_output_size=8, dropout=(0, 0, 0)):
        """
            Args:
                embed_vecs - tensor; pretrained embedding vectors.
                hidden_size - int; dimension of the GRU hidden layers.
                attn_output_size - int; number of attention outputs.
                dropout - tuple(float, float, float); attention applied
                after the the embedding, after GRU and between the linear
                layers in the head.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_output_size = attn_output_size
        self.embedding = nn.Embedding.from_pretrained(embed_vecs)

        self.gru = nn.GRU(input_size=embed_vecs.size()[1], hidden_size=hidden_size,
                          num_layers=1, bidirectional=True, batch_first=True)
        self.emb_dropout = nn.Dropout(p=dropout[0])
        self.dropout = nn.Dropout(p=dropout[1])
        self.attention = SelfAttention(2*self.hidden_size, attn_output_size)
        self.head = nn.Sequential(
            nn.Linear(2*self.attn_output_size*self.hidden_size, self.attn_output_size*self.hidden_size),
            nn.Dropout(p=dropout[2]),
            nn.ReLU(),
            nn.Linear(self.attn_output_size*self.hidden_size, 2))

    def forward(self, batch):
        """
            Args:
                batch - tuple(batch, lengths);
        """
        batch, lengths = batch
        batch_dim, _ = batch.shape

        embedded = self.emb_dropout(self.embedding(batch)) # B x L x E

        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs_packed, hiddens = self.gru(embedded_packed)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        outputs = self.dropout(outputs) # B x L x 2H

        attn_weights = self.attention(outputs, lengths) # B x A x L
        attn_output = torch.bmm(attn_weights, outputs).view(batch_dim, -1) # B x A*2H

        logging.debug('batch shape : {}'.format(batch.shape))
        logging.debug('embedding shape : {}'.format(embedded.shape))
        logging.debug('hiddens shape : {}'.format(hiddens.shape))
        logging.debug('outputs shape : {}'.format(outputs.shape))
        logging.debug('attn_weights shape : {}'.format(attn_weights.shape))
        logging.debug('attn_output shape : {}'.format(attn_output.shape))

        return self.head(attn_output)


class Hook():
    """
        Registers a forward hook in a specified module.
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output


class CrossEntropyWP(nn.Module):
    """
        A loss function which combines cross entropy loss and
        a penalty term to reduce redundancy among the attention
        outputs.
    """
    def __init__(self, attn_size, hook, penalty=0):
        """
            Args:
                attn_size - int; number of attention outputs.
                hook - Hook; a forward hook attached to the self attention layer.
                penalty - float; a multiplicative factor for the penalty term.
        """
        super().__init__()
        self.crossentropy = nn.CrossEntropyLoss()
        self.eye = torch.eye(attn_size).to(device)
        self.hook = hook
        self.penalty = penalty

    def forward(self, model_output, label):
        loss = self.crossentropy(model_output, label)

        attn_matrix = self.hook.output
        logging.debug('attn_matrix shape : {}'.format(attn_matrix.size()))
        penalty_term = torch.bmm(attn_matrix, attn_matrix.transpose(1, 2))
        penalty_term = torch.norm(penalty_term - self.eye, dim=(1, 2)).pow(2).mean()

        loss += self.penalty * penalty_term
        return loss


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
    model = GRUSelfAttention(emb_weights.clone(), hidden_size=32, attn_output_size=8,
                             dropout=(0.5, 0.5, 0.5)).to(device)
    loss_fn = CrossEntropyWP(attn_size=8, hook=Hook(model.attention), penalty=0.03).to(device)
    optimiser = AdamW(model.parameters(), weight_decay=1e-3)
    cycle_steps = (len(ds_train)//64 + 1)
    scheduler = CyclicLRDecay(optimiser, 2e-5, 1e-3, cycle_steps, 0.1, gamma_factor=0.85)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=2, bs=bs, grad_clip=(0.15, 1.5))

    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    learner.train(epochs=6, bs=bs, grad_clip=(0.15, 1.5))

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
