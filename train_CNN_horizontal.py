import logging
import argparse
import random
import os
import wget
import zipfile
import re
from functools import partial
import torch
import torch.nn as nn
import torchtext.data as data
import torchtext.vocab as vocab
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from utils import learner, validate, cross_validate
from transformers import BertTokenizer, BertModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self, out_channels, kernel_heights, stride, padding, dropout, emb_weights):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.embedding_length = emb_weights.shape[1]

        self.word_embeddings = nn.Embedding.from_pretrained(emb_weights)
        self.conv1 = nn.Conv2d(1, out_channels, (kernel_heights[0], self.embedding_length), stride, (padding[0], 0))
        self.conv2 = nn.Conv2d(1, out_channels, (kernel_heights[1], self.embedding_length), stride, (padding[1], 0))
        self.conv3 = nn.Conv2d(1, out_channels, (kernel_heights[2], self.embedding_length), stride, (padding[2], 0))
        # self.conv4 = nn.Conv2d(1, out_channels, (kernel_heights[3], self.embedding_length), stride, (padding[3], 0))
        self.dropout0 = nn.Dropout(p=dropout[0])
        self.dropout1 = nn.Dropout(p=dropout[1])
        self.relu = nn.ReLU()
        self.head = nn.Linear(len(kernel_heights)*out_channels, 2)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = self.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = nn.functional.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, batch):
        input = self.word_embeddings(batch).unsqueeze(1)

        max_out = []
        max_out.append(self.conv_block(self.dropout0(input), self.conv1))
        max_out.append(self.conv_block(self.dropout0(input), self.conv2))
        max_out.append(self.conv_block(self.dropout0(input), self.conv3))

        all_out = torch.cat(max_out, dim=1)

        return self.head(self.dropout1(all_out))


def training(model, loss_fn, ds_train, ds_val):
    optimiser = Adam(model.parameters())
    scheduler_fn = partial(OneCycleLR, pct_start=0.3, max_lr=5e-4, steps_per_epoch=len(ds_train)//8+1, epochs=1)
    logging.info('Training, \'one cycle\' lr=5e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=10, bs=8, scheduler_fn=scheduler_fn)
    logging.info('Unfreezing the embedding layer')
    model.word_embeddings.weight.requires_grad_(True)
    scheduler_fn = partial(OneCycleLR, pct_start=0.3, max_lr=1e-4, steps_per_epoch=len(ds_train)//8+1, epochs=1)
    logging.info('Training, \'one cycle\' lr=1e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8, scheduler_fn=scheduler_fn)


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
    if args.embedding == 'glove':
        from get_dataset import get_dataset
    else:
        from get_dataset import get_dataset_bert as get_dataset
    dataset, emb_weights = get_dataset()

    if args.randomseed is not None:
        random.seed(args.randomseed)

    # create model and loss function
    model = CNN(out_channels=16, kernel_heights=(1, 3, 5), stride=1, padding=(0, 1, 2),
                dropout=(0.3, 0.3), emb_weights=emb_weights.clone())
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
        logging.info('Model saved to: models/model_CNN.pt')
        torch.save(model.state_dict(), 'models/model_CNN.pt')


if __name__ == '__main__':
    main()
