from utils import validate, learner
from get_dataset import get_dataset
import random
import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.data import BucketIterator
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logger
import logging
logging.basicConfig(filename='log.txt',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %H:%M:%S')  


dataset, emb_weights = get_dataset()
vocab_size = len(emb_weights)
embed_size = emb_weights.shape[1] # 300
dataset.fields['review'].include_lengths = True
# the above line adds a tensor with lengths of reviews in a batch, so that we can use padded sequences

# random.seed(43)
ds_train, ds_val, ds_test = dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.getstate())  


class GRUmodel(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_vecs=None, hidden_size=512,
                num_layers=1, dropout=(0, 0), bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if embed_vecs is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_vecs)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout[1],
                          bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(p=dropout[0])
        self.head = nn.Linear(3*self.num_directions*self.hidden_size, 2)

        
    def forward(self, batch):
        batch, lengths = batch
        batch_dim, _ = batch.shape
        
        embedded = self.dropout(self.embedding(batch))
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        outputs_packed, hiddens = self.gru(embedded_packed)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        
        last_hidden = hiddens.view(self.num_layers, self.num_directions, batch_dim, self.hidden_size)[-1,:,:,:]
        hidden_concat = last_hidden.transpose(1,0).reshape(batch_dim, self.num_directions*self.hidden_size)
        
        avg_pool = torch.sum(outputs, dim=1)/lengths.unsqueeze(1)
        max_pool = torch.cat([sample[:length].max(dim=0)[0].unsqueeze(0) for sample, length in zip(outputs, lengths)], dim=0)
        
        logging.debug('batch shape : {}'.format(batch.shape))
        logging.debug('embedding shape : {}'.format(embedded.shape))
        logging.debug('hiddens shape : {}'.format(hiddens.shape))
        logging.debug('outputs shape : {}'.format(outputs.shape))
        logging.debug('avg_pool shape : {}'.format(avg_pool.shape))
        logging.debug('max_pool shape : {}'.format(max_pool.shape))
        logging.debug('hidden_concat shape : {}'.format(hidden_concat.shape))
        
        return self.head(torch.cat([hidden_concat, avg_pool, max_pool], dim=1))

    

logging.info('Initialising the GRU model with concat pooling layer (with the embedding layer frozen).')
model = GRUmodel(vocab_size, embed_size, emb_weights.clone(), num_layers=2, hidden_size=32,
                 bidirectional=True, dropout=(0.3, 0.3)).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimiser = Adam(model.parameters(), lr=3e-4)

logging.info('Training, lr=3e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)
optimiser.param_groups[0]['lr'] = 1e-4
logging.info('Training, lr=1e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)

logging.info('Unfreezing the embedding layer')
model.embedding.weight.requires_grad_(True);
optimiser = Adam(model.parameters(), lr=1e-4)
logging.info('Training, lr=1e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)

if not os.path.exists('models'):
    os.makedirs('models')
logging.info('Model saved to: models/model_GRU_concatpool.pt')
torch.save(model.state_dict(), 'models/model_GRU_concatpool.pt')

logging.info('--- Evaluating on the test set ---')
logging.info('Test loss : {:.5f}, test accuracy : {:.03f}'.format(*validate(ds_test, loss_fn, model)))



class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxQ]
        # Values = [BxTxQ]
        # Outputs = a:[TxB], lin_comb:[BxQ]

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxQ] -> [BxQxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxQxT] -> [Bx1xT]
        energy = self.softmax(energy.mul_(self.scale)) # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxQ] -> [BxQ]
        return energy, linear_combination
    
class GRU_Attention_model(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_vecs=None, hidden_size=512,
                num_layers=1, dropout=(0, 0), bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if embed_vecs is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_vecs)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout[1],
                          bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(p=dropout[0])
        self.attention = Attention(self.hidden_size*self.num_directions)
        self.head = nn.Linear(self.num_directions*self.hidden_size, 2)

        
    def forward(self, batch):
        batch, lengths = batch
        batch_dim, _ = batch.shape
        
        embedded = self.dropout(self.embedding(batch))
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        outputs_packed, hiddens = self.gru(embedded_packed)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        
        last_hidden = hiddens.view(self.num_layers, self.num_directions, batch_dim, self.hidden_size)[-1,:,:,:]
        hidden_concat = last_hidden.transpose(1,0).reshape(batch_dim, self.num_directions*self.hidden_size)
        
        energy, attended_output = self.attention(hidden_concat, outputs, outputs)
        
        logging.debug('batch shape : {}'.format(batch.shape))
        logging.debug('embedding shape : {}'.format(embedded.shape))
        logging.debug('hiddens shape : {}'.format(hiddens.shape))
        logging.debug('outputs shape : {}'.format(outputs.shape))
        logging.debug('hidden_concat shape : {}'.format(hidden_concat.shape))
        logging.debug('attended_output shape : {}'.format(attended_output.shape))
        
        return self.head(attended_output)
    

logging.info('Initialising the GRU model with dot product attention layer (with the embedding layer frozen).')
model = GRU_Attention_model(vocab_size, embed_size, emb_weights.clone(),num_layers=2, hidden_size=32,
                            bidirectional=True, dropout=(0.3, 0.3)).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimiser = Adam(model.parameters(), lr=3e-4)

logging.info('Training, lr=3e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)
optimiser.param_groups[0]['lr'] = 1e-4
logging.info('Training, lr=1e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)

logging.info('Unfreezing the embedding layer')
model.embedding.weight.requires_grad_(True);
optimiser = Adam(model.parameters(), lr=1e-4)
logging.info('Training, lr=1e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)

if not os.path.exists('models'):
    os.makedirs('models')
logging.info('Model saved to: models/model_GRU_attention.pt')
torch.save(model.state_dict(), 'models/model_GRU_attention.pt')

logging.info('--- Evaluating on the test set ---')
logging.info('Test loss : {:.5f}, test accuracy : {:.03f}'.format(*validate(ds_test, loss_fn, model)))

