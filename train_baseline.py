import utils
from get_dataset import get_dataset
import random
import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.data import Iterator
from importlib import reload

# Setup logger
import logging
logging.basicConfig(filename='log.txt',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s : %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset, emb_weights = get_dataset()
random.seed(43)
ds_train, ds_val, ds_test = dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.getstate())

class Baseline_model(nn.Module):
  def __init__(self, vocab_size, embed_dim, embed_vecs=None):
    super().__init__()
    self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
    if embed_vecs is not None:
      self.embedding = nn.EmbeddingBag.from_pretrained(embed_vecs)
    else:
      self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
    self.fc1 = nn.Linear(embed_dim, embed_dim//2)
    self.fc2 = nn.Linear(embed_dim//2, 2)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.embedding(x)
    out = self.relu(self.fc1(out))
    return self.fc2(out)

def learner(model, loss_fn, optimiser, epochs=1, device=device):
  start_time = time.time()
  for epoch in range(epochs):
    
    total_loss = 0
    for i, batch in enumerate(Iterator(ds_train, 1, shuffle=True, device=device), 1):
      optimiser.zero_grad()

      output = model(batch.review)
      loss = loss_fn(output, batch.label)
      total_loss += loss.item()

      loss.backward()
      optimiser.step()

      if not i % (len(ds_train)//3):
        avg_loss = total_loss / (len(ds_train)//3)
        val_loss, val_accuracy = utils.validate(ds_val, loss_fn, model)
        logging.info('Epoch : {}, batch : {}, train_loss = {:.4f}, val_loss = {:.4f}, val_accuracy : {:.3f}'
                     .format(epoch + 1, i,avg_loss, val_loss, val_accuracy))
        total_loss = 0

vocab_size = len(emb_weights)
embed_size = 300

logging.info('Initialising the model with the embedding layer frozen.')
model = Baseline_model(vocab_size, embed_size, emb_weights).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimiser = Adam(model.parameters(), lr=2e-4)

logging.info('Training, lr=2e-4')
learner(model, loss_fn, optimiser, epochs=5)
if not os.path.exists('models'):
    os.makedirs('models')
    
logging.info('Model saved to: models/model_baseline.pt')
torch.save(model.state_dict(), 'models/model_baseline.pt')

logging.info('Unfreezing the embedding layer')
model.embedding.weight.requires_grad_(True);
optimiser = Adam(model.parameters(), lr=1e-4)

logging.info('Fine-tuning, lr=1e-4')
learner(model, loss_fn, optimiser, epochs=5)

logging.info('Model saved to: models/model_baseline_fine.pt')
torch.save(model.state_dict(), 'models/model_baseline_fine.pt')

logging.info('--- Evaluating on the test set ---')
logging.info('Test loss : {:.5f}, test accuracy : {:.03f}'.format(*utils.validate(ds_test, loss_fn, model)))
