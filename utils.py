import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.data import Iterator
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(ds, loss_fn, model, device=device):
  with torch.no_grad():
    size = len(ds)
    predictions = []
    gt = list(ds.label)
    loss = 0
    for i, batch in enumerate(Iterator(ds, 1, shuffle=False, device=device)):
      output = model(batch.review)
      predictions.extend(output.argmax(dim=1).tolist())
      loss += loss_fn(output, batch.label).item()
    avg_loss = loss/size
  
  accuracy = np.mean(np.array(predictions) == np.array(gt))
  return avg_loss, accuracy


def lr_finder(model, dataset, optimiser, loss_fn, lr_range=[1e-6, 1e0], avg_over_batches=200, device=device):
  lr_list = np.logspace(np.log10(lr_range[0]),
                        np.log10(lr_range[1]),
                        np.int(np.log10(lr_range[1]/lr_range[0])*10))
  initial_state_dict = copy.deepcopy(model.state_dict())
  losses = []
  learning_rates = []
  for param_group in optimiser.param_groups:
      param_group['lr'] = lr_list[0]

  for i, batch in enumerate(Iterator(dataset, 1, shuffle=True, device=device, repeat=True), 1):
    optimiser.zero_grad()

    output = model(batch.review)
    loss = loss_fn(output, batch.label)
    losses.append(loss.item())
    learning_rates.append(optimiser.param_groups[0]['lr'])

    loss.backward()
    optimiser.step()

    if i == 1:
      initial_loss = loss.item()
    elif loss.item() > 100*initial_loss:
      break
    elif (i // avg_over_batches) == len(lr_list):
      break
    elif not i % avg_over_batches:
      for param_group in optimiser.param_groups:
        param_group['lr'] = lr_list[i // avg_over_batches]
  
  model.load_state_dict(initial_state_dict)
  sns.lineplot(learning_rates, losses)
  plt.xscale('log')