import logging
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import validate, learner, cross_validate
from get_dataset import get_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRUmodel_concatpool(nn.Module):
    def __init__(self, embed_vecs, hidden_size=512,
                 num_layers=1, dropout=(0, 0), bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_vecs.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embed_vecs)
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout[1],
                          bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(p=dropout[0])
        self.head = nn.Linear(3*self.num_directions*self.hidden_size, 2)

    def forward(self, batch):
        batch, lengths = batch
        batch_dim, _ = batch.shape

        embedded = self.dropout(self.embedding(batch))  # (B x E)
        embedded_packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)

        outputs_packed, hiddens = self.gru(embedded_packed)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        # outputs : (B x L x H)
        # hiddens : (L x B x H)

        last_hidden = hiddens.view(self.num_layers, self.num_directions, batch_dim, self.hidden_size)[-1, :, :, :]
        hidden_concat = last_hidden.transpose(1, 0).reshape(batch_dim, self.num_directions*self.hidden_size)  # (B x H)

        avg_pool = torch.sum(outputs, dim=1)/lengths.unsqueeze(1)  # (B x H)
        max_pool = torch.cat([sample[:length].max(dim=0)[0].unsqueeze(0) for sample, length in zip(outputs, lengths)], dim=0)  # (B x H)

        logging.debug('batch shape : {}'.format(batch.shape))
        logging.debug('embedding shape : {}'.format(embedded.shape))
        logging.debug('hiddens shape : {}'.format(hiddens.shape))
        logging.debug('outputs shape : {}'.format(outputs.shape))
        logging.debug('avg_pool shape : {}'.format(avg_pool.shape))
        logging.debug('max_pool shape : {}'.format(max_pool.shape))
        logging.debug('hidden_concat shape : {}'.format(hidden_concat.shape))

        return self.head(torch.cat([hidden_concat, avg_pool, max_pool], dim=1))


def training(model, loss_fn, ds_train, ds_val):
    optimiser = Adam(model.parameters(), lr=3e-4)
    logging.info('Training, lr=3e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)
    optimiser.param_groups[0]['lr'] = 1e-4
    logging.info('Training, lr=1e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)
    logging.info('Unfreezing the embedding layer')
    model.embedding.weight.requires_grad_(True)
    logging.info('Training, lr=1e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)


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
    model = GRUmodel_concatpool(emb_weights.clone(), num_layers=2, hidden_size=32,
                                bidirectional=True, dropout=(0.3, 0.3)).to(device)
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
        logging.info('Model saved to: models/model_GRU_concatpool.pt')
        torch.save(model.state_dict(), 'models/model_GRU_concatpool.pt')


if __name__ == '__main__':
    main()
