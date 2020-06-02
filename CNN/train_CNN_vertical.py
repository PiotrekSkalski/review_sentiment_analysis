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


# class CNN(nn.Module):
#     def __init__(self, out_channels, kernel_heights, stride, padding, dropout, emb_weights):
#         super().__init__()

#         self.out_channels = out_channels
#         self.kernel_heights = kernel_heights
#         self.stride = stride
#         self.padding = padding
#         self.emb_dim = emb_weights.shape[1]

#         self.word_embeddings = nn.Embedding.from_pretrained(emb_weights)
#         self.conv1 = nn.Conv2d(1, 4*out_channels, (kernel_heights[0], self.emb_dim),
#                                stride, (padding[0], 0))
#         self.conv2 = nn.Conv2d(4*out_channels, 2*out_channels, (kernel_heights[1], 1),
#                                stride, (padding[1], 0))
#         self.conv3 = nn.Conv2d(6*out_channels, 3*out_channels, (kernel_heights[2], 1),
#                                stride, (padding[2], 0))
#         self.conv4 = nn.Conv2d(9*out_channels, 4*out_channels, (kernel_heights[3], 1),
#                                stride, (padding[3], 0))
#         self.conv5 = nn.Conv2d(13*out_channels, 6*out_channels, (kernel_heights[4], 1),
#                                stride, (padding[4], 0))
#         self.dropout0 = nn.Dropout(p=dropout[0])
#         self.dropout1 = nn.Dropout(p=dropout[1])
#         self.relu = nn.ReLU()
#         self.head = nn.Linear(19*out_channels, 2)

#     def forward(self, batch):
#         input = self.word_embeddings(batch).unsqueeze(1)
#         # input : (B x 1 x L x E)

#         out = []
#         out.append(self.relu(self.conv1(self.dropout0(input))))  # (B x 4*C x L x 1)
#         out.append(self.relu(self.conv2(self.dropout0(out[-1]))))  # (B x 2*C x L x 1)
#         skip1 = torch.cat([out[-1], out[-2]], dim=1)  # (B x 6*C x L x 1)
#         out.append(self.relu(self.conv3(self.dropout0(skip1))))  # (B x 3*C x L x 1)
#         skip2 = torch.cat([out[-1], skip1], dim=1)  # (B x 9*C x L x 1)
#         out.append(self.relu(self.conv4(self.dropout0(skip2))))  # (B x 4*C x L x 1)
#         skip3 = torch.cat([out[-1], skip2], dim=1)  # (B x 13*C x L x 1)
#         out.append(self.relu(self.conv5(self.dropout0(skip3))))  # (B x 6*C x L x 1)
#         skip4 = torch.cat([out[-1], skip3], dim=1)  # (B x 19*C x L x 1)

#         max_out = nn.functional.max_pool1d(skip4.squeeze(3), skip4.size()[2]).squeeze(2)  # (B x 27*C)

#         return self.head(self.dropout1(max_out))


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
        self.conv2 = nn.Conv2d(out_channels, 2*out_channels, (kernel_heights[1], 1), stride, (padding[1], 0))
        self.conv3 = nn.Conv2d(2*out_channels, 4*out_channels, (kernel_heights[2], 1), stride, (padding[2], 0))
        self.dropout0 = nn.Dropout(p=dropout[0])
        self.dropout1 = nn.Dropout(p=dropout[1])
        self.relu = nn.ReLU()
        self.head = nn.Linear(4*out_channels, 2)

    def forward(self, batch):
        input = self.word_embeddings(batch).unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)

        out = self.relu(self.conv1(self.dropout0(input)))
        out = self.relu(self.conv2(self.dropout0(out)))
        out = self.relu(self.conv3(self.dropout0(out)))
        max_out = max_out = nn.functional.max_pool1d(out.squeeze(3), out.size()[2]).squeeze(2)

        return self.head(self.dropout1(max_out))


def training(model, loss_fn, ds_train, ds_val):
    optimiser = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    cycle_steps = 2*(len(ds_train)//32 + 1)
    scheduler = CyclicLRDecay(optimiser, 1e-4, 5e-3, cycle_steps, 0.25, gamma_factor=0.75)
    learner = Learner(model, loss_fn, optimiser, scheduler, ds_train, ds_val, device)

    logging.info('Training')
    learner.train(epochs=10, bs=32, grad_clip=(0.2, 1.5))

    logging.info('Unfreezing the embedding layer')
    model.word_embeddings.weight.requires_grad_(True)
    learner.train(epochs=6, bs=32, grad_clip=(0.2, 1.5))


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
                        help='number of folds, default=10.')
    args = parser.parse_args()

    # get dataset
    dataset, emb_weights = get_dataset()

    if args.randomseed is not None:
        random.seed(args.randomseed)

    # create model and loss function
    model = CNN(out_channels=32, kernel_heights=(1, 3, 5), stride=1,
                padding=(0, 1, 2), dropout=(0.3, 0.35),
                emb_weights=emb_weights.clone()).to(device)
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
        logging.info('Model saved to: models/model_CNN_vertical.pt')
        torch.save(model.state_dict(), 'models/model_CNN_vertical.pt')


if __name__ == '__main__':
    main()
