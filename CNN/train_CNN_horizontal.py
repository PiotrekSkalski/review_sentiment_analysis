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
from utils import validate, cross_validate, get_dataset
from learner import Learner
from schedulers import CyclicLRDecay


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
        self.conv4 = nn.Conv2d(1, out_channels, (kernel_heights[3], self.embedding_length), stride, (padding[3], 0))
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
        input = self.dropout0(input)

        max_out = []
        max_out.append(self.conv_block(input, self.conv1))
        max_out.append(self.conv_block(input, self.conv2))
        max_out.append(self.conv_block(input, self.conv3))
        max_out.append(self.conv_block(input, self.conv4))

        all_out = torch.cat(max_out, dim=1)

        return self.head(self.dropout1(all_out))


def training(model, loss_fn, ds_train, ds_val):
    optimiser = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    cycle_steps = 2*(len(ds_train)//32 + 1)
    scheduler = CyclicLRDecay(optimiser, 3e-5, 5e-3, cycle_steps, 0.25, gamma_factor=0.75)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=10, bs=32, grad_clip=(0.17, 1.6))

    logging.info('Unfreezing the embedding layer')
    model.word_embeddings.weight.requires_grad_(True)
    learner.train(epochs=6, bs=32, grad_clip=(0.17, 1.6))


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

    if args.randomseed is not None:
        random.seed(args.randomseed)

    # create model and loss function
    model = CNN(out_channels=8, kernel_heights=(1, 3, 5, 7), stride=1, padding=(0, 1, 2, 3),
                dropout=(0.45, 0.5), emb_weights=emb_weights.clone())
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
        logging.info('Model saved to: models/model_CNN_horizontal.pt')
        torch.save(model.state_dict(), 'models/model_CNN_horizontal.pt')


if __name__ == '__main__':
    main()
