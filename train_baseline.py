from utils import validate, learner
from get_dataset import get_dataset
import random
import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from importlib import reload

# Setup logger
import logging
logging.basicConfig(filename='log.txt',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset, emb_weights = get_dataset()
vocab_size = len(emb_weights)
embed_size = emb_weights.shape[1] # 300
dataset.fields['review'].include_lengths = True
# the above line adds a tensor with lengths of reviews in a batch, so that we can pass batches to embeddingbag

# random.seed(43)
ds_train, ds_val, ds_test = dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.getstate())

class Baseline_model(nn.Module):
  """
     Basic baseline model based on a bag of embeddings.
     
     Args:
         vocab_size - int; size of the dataset vocabulary .
         embed_dim - int; size of the embedding vectors.
         embed_vecs - tensor; pretrained word embeddings (optional).
  """
  def __init__(self, vocab_size, embed_dim, embed_vecs=None):
    super().__init__()
    self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
    if embed_vecs is not None:
      self.embedding = nn.EmbeddingBag.from_pretrained(embed_vecs)
    else:
      self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
    self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim//2),
                              nn.ReLU(),
                              nn.Linear(embed_dim//2, 2))

  def forward(self, batch):
    batch, lengths = batch
    
    batch_concat = torch.cat([review[:length] for review, length in zip(batch, lengths)], dim=0)
    offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths[:-1], dim=0)])
    
    out = self.embedding(batch_concat, offsets)
    return self.head(out)



logging.info('Initialising the model with the embedding layer frozen.')
model = Baseline_model(vocab_size, embed_size, emb_weights).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimiser = Adam(model.parameters(), lr=2e-4)

logging.info('Training, lr=2e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=10, bs=8)
if not os.path.exists('models'):
    os.makedirs('models')
    
logging.info('Model saved to: models/model_baseline.pt')
torch.save(model.state_dict(), 'models/model_baseline.pt')

logging.info('Unfreezing the embedding layer')
model.embedding.weight.requires_grad_(True);
optimiser = Adam(model.parameters(), lr=1e-4)

logging.info('Fine-tuning, lr=1e-4')
learner(model, loss_fn, optimiser, ds_train, ds_val, epochs=10, bs=8)

logging.info('Model saved to: models/model_baseline_fine.pt')
torch.save(model.state_dict(), 'models/model_baseline_fine.pt')

logging.info('--- Evaluating on the test set ---')
logging.info('Test loss : {:.5f}, test accuracy : {:.03f}'.format(*validate(ds_test, loss_fn, model, bs=8)))
