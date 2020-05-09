import logging
import argparse
import random
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import learner, validate, cross_validate
from get_dataset import get_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self, out_channels, kernel_heights, stride, padding, dropout, emb_weights):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.emb_dim = emb_weights.shape[1]

        self.word_embeddings = nn.Embedding.from_pretrained(emb_weights)
        self.conv1 = nn.Conv2d(1, 4*out_channels, (kernel_heights[0], self.emb_dim),
                               stride, (padding[0], 0))
        self.conv2 = nn.Conv2d(4*out_channels, 2*out_channels, (kernel_heights[1], 1),
                               stride, (padding[1], 0))
        self.conv3 = nn.Conv2d(6*out_channels, 3*out_channels, (kernel_heights[2], 1),
                               stride, (padding[2], 0))
        self.conv4 = nn.Conv2d(9*out_channels, 4*out_channels, (kernel_heights[3], 1),
                               stride, (padding[3], 0))
        self.conv5 = nn.Conv2d(13*out_channels, 6*out_channels, (kernel_heights[4], 1),
                               stride, (padding[4], 0))
        self.dropout0 = nn.Dropout(p=dropout[0])
        self.dropout1 = nn.Dropout(p=dropout[1])
        self.relu = nn.ReLU()
        self.head = nn.Linear(19*out_channels, 2)

    def forward(self, batch):
        input = self.word_embeddings(batch).unsqueeze(1)
        # input : (B x 1 x L x E)

        out = []
        out.append(self.relu(self.conv1(self.dropout0(input))))  # (B x 4*C x L x 1)
        out.append(self.relu(self.conv2(self.dropout0(out[-1]))))  # (B x 2*C x L x 1)
        skip1 = torch.cat([out[-1], out[-2]], dim=1)  # (B x 6*C x L x 1)
        out.append(self.relu(self.conv3(self.dropout0(skip1))))  # (B x 3*C x L x 1)
        skip2 = torch.cat([out[-1], skip1], dim=1)  # (B x 9*C x L x 1)
        out.append(self.relu(self.conv4(self.dropout0(skip2))))  # (B x 4*C x L x 1)
        skip3 = torch.cat([out[-1], skip2], dim=1)  # (B x 13*C x L x 1)
        out.append(self.relu(self.conv5(self.dropout0(skip3))))  # (B x 6*C x L x 1)
        skip4 = torch.cat([out[-1], skip3], dim=1)  # (B x 19*C x L x 1)

        max_out = nn.functional.max_pool1d(skip4.squeeze(3), skip4.size()[2]).squeeze(2)  # (B x 27*C)

        return self.head(self.dropout1(max_out))


def training(model, loss_fn, ds_train, ds_val):
    optimiser = Adam(model.parameters(), lr=3e-4)
    logging.info('Training, lr=3e-4')
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=8, bs=8)
    logging.info('Training, lr=1e-4')
    optimiser.param_groups[0]['lr'] = 1e-4
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=3, bs=8)
    logging.info('Unfreezing the embedding layer')
    model.word_embeddings.weight.requires_grad_(True)
    logging.info('Training, lr=1e-4')
    optimiser.param_groups[0]['lr'] = 1e-4
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=3, bs=8)
    logging.info('Training, lr=5e-5')
    optimiser.param_groups[0]['lr'] = 5e-5
    learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=3, bs=8)


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
    model = CNN(out_channels=8, kernel_heights=(1, 3, 5, 7, 9), stride=1,
                padding=(0, 1, 2, 3, 4), dropout=(0.4, 0.4),
                emb_weights=emb_weights.clone()).to(device)
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
