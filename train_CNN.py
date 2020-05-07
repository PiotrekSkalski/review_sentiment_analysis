from utils import lr_finder, learner, validate, MyIterator
from get_dataset import get_dataset
import random
import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logger
import logging
logging.basicConfig(filename='log.txt',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %H:%M:%S')  

dataset, emb_weights = get_dataset()
# random.seed(43)
ds_train, ds_val, ds_test = dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.getstate())

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
        # input : (B x 1 x L x E)
        
        out = self.relu(self.conv1(self.dropout0(input))) # (B x C x L x 1)
        out = self.relu(self.conv2(self.dropout0(out))) # (B x 2*C x L x 1)
        out = self.relu(self.conv3(self.dropout0(out))) # (B x 4*C x L x 1)
        max_out = max_out = nn.functional.max_pool1d(out.squeeze(3), out.size()[2]).squeeze(2) # (B x 4*C)

        return self.head(self.dropout1(max_out))
    

logging.info('Initialising the CNN model (with the embedding layer frozen).')
model = CNN(out_channels=16, kernel_heights=(1, 3, 5, 7), stride=1, padding=(0, 1, 2, 3), dropout=(0.4, 0.4), emb_weights=emb_weights.clone()).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

logging.info('Training, lr=3e-4')
optimiser = Adam(model.parameters(), lr=3e-4)
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=8, bs=8)

logging.info('Training, lr=1e-4')
optimiser.param_groups[0]['lr'] = 1e-4
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=3, bs=8)

logging.info('Unfreezing the embedding layer')
model.word_embeddings.weight.requires_grad_(True);
logging.info('Training, lr=1e-4')
optimiser.param_groups[0]['lr'] = 1e-4
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=3, bs=8)

logging.info('Training, lr=5e-5')
optimiser.param_groups[0]['lr'] = 5e-5
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=3, bs=8)

if not os.path.exists('models'):
    os.makedirs('models')
logging.info('Model saved to: models/model_CNN.pt')
torch.save(model.state_dict(), 'models/model_CNN.pt')

logging.info('--- Evaluating on the test set ---')
logging.info('Test loss : {:.5f}, test accuracy : {:.03f}'.format(*validate(ds_test, loss_fn, model)))