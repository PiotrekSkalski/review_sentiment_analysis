from utils import validate, learner
from get_dataset import get_dataset
import random
import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
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



class SelfAttention(nn.Module):
    def __init__(self, query_dim, n_outputs, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(query_dim, query_dim//2)
        self.W2 = nn.Linear(query_dim//2, n_outputs)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query):
        attn_weights = self.W2(self.dropout(torch.tanh(self.W1(query))))
        attn_weights = attn_weights.permute(0,2,1)
        
        return self.softmax(attn_weights)
    
class GRU_SelfAttention_model(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_vecs=None, hidden_size=512,
                num_layers=1, attn_output_size=8, dropout=(0.2, 0.2), bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attn_output_size = attn_output_size
        if embed_vecs is not None:
            self.embedding = nn.Embedding.from_pretrained(embed_vecs)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout[1],
                          bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(p=dropout[0])
        self.attention = SelfAttention(self.num_directions*self.hidden_size, self.attn_output_size, dropout[0])
        self.head = nn.Linear(self.attn_output_size*self.num_directions*self.hidden_size, 2)

        
    def forward(self, batch):
        batch, lengths = batch
        batch_dim, _ = batch.shape
        
        embedded = self.dropout(self.embedding(batch))
        embedded_packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        outputs_packed, hiddens = self.gru(embedded_packed)
        
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)

        attn_weights = self.attention(self.dropout(outputs))
        attn_output = torch.bmm(attn_weights, outputs).view(batch_dim, -1)
        
        logging.debug('batch shape : {}'.format(batch.shape))
        logging.debug('embedding shape : {}'.format(embedded.shape))
        logging.debug('hiddens shape : {}'.format(hiddens.shape))
        logging.debug('outputs shape : {}'.format(outputs.shape))
        logging.debug('attn_weights shape : {}'.format(attn_weights.shape))
        logging.debug('attn_output shape : {}'.format(attn_output.shape))
        
        return self.head(self.dropout(attn_output))
    

    
logging.info('Initialising the GRU model with dot product attention layer (with the embedding layer frozen).')
model = GRU_SelfAttention_model(vocab_size, embed_size, emb_weights.clone(), bidirectional=True,
                                num_layers=2, hidden_size=32, attn_output_size=8, dropout=(0.4, 0.4)).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimiser = Adam(model.parameters(), lr=3e-4)

logging.info('Training, lr=3e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=8, bs=8)
optimiser.param_groups[0]['lr'] = 1e-4
logging.info('Training, lr=1e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=3, bs=8)

logging.info('Unfreezing the embedding layer')
model.embedding.weight.requires_grad_(True);
optimiser.param_groups[0]['lr'] = 1e-4
logging.info('Training, lr=1e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)

optimiser.param_groups[0]['lr'] = 5e-5
logging.info('Training, lr=5e-5')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=5, bs=8)

if not os.path.exists('models'):
    os.makedirs('models')
logging.info('Model saved to: models/model_GRU_selfattention.pt')
torch.save(model.state_dict(), 'models/model_GRU_selfattention.pt')

logging.info('--- Evaluating on the test set ---')
logging.info('Test loss : {:.5f}, test accuracy : {:.03f}'.format(*validate(ds_test, loss_fn, model)))



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

