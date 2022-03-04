#!/usr/bin/env python
# coding: utf-8

# # Assignment 2

# ## Initialization

# In[ ]:


#@title Mount your Google Drive
# If you run this notebook locally or on a cluster (i.e. not on Google Colab)
# you can delete this cell which is specific to Google Colab. You may also
# change the paths for data/logs in Arguments below.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


#@title Link your assignment folder & install requirements
#@markdown Enter the path to the assignment folder in your Google Drive
# If you run this notebook locally or on a cluster (i.e. not on Google Colab)
# you can delete this cell which is specific to Google Colab. You may also
# change the paths for data/logs in Arguments below.
import sys
import os
import shutil
import warnings

folder = "" #@param {type:"string"}
get_ipython().system('ln -Ts "$folder" /content/assignment 2> /dev/null')

# Add the assignment folder to Python path
if '/content/assignment' not in sys.path:
  sys.path.insert(0, '/content/assignment')

# Install requirements
get_ipython().system('pip install -qr /content/assignment/requirements.txt')

# Check if CUDA is available
import torch
if not torch.cuda.is_available():
  warnings.warn('CUDA is not available.')


# ### Running on GPU
# For this assignment, it will be necessary to run your experiments on GPU. To make sure the notebook is running on GPU, you can change the notebook settings with
# * (EN) `Edit > Notebook Settings`
# * (FR) `Modifier > Paramètres du notebook`
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import urllib.request

from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm

from lstm_solution import LSTM
from utils.wikitext2 import Wikitext2
from utils.torch_utils import seed_experiment, to_device
from utils.data_utils import save_logs
from run_exp import train, evaluate


# ## Experiments

# Below we define a few default arguments to get you started with your experiments. You are encouraged to modify the function `main()`, as well as these arguments, to fit your needs (e.g. changing hyperparameters, the optimizer, adding regularization, adding logs).

# In[ ]:


@dataclass
class Arguments:
  # Data
  data_folder: str = '/content/assignment/data'
  batch_size: int = 16

  # Model
  model: str = 'lstm'  # [lstm, gpt1]
  embeddings: str = '/content/assignment/data/embeddings.npz'
  layers: int = 1

  # Optimization
  optimizer: str = 'adamw'  # [sgd, momentum, adam, adamw]
  epochs: int = 10
  lr: float = 1e-3
  momentum: float = 0.9
  weight_decay: float = 5e-4

  # Experiment
  exp_id: str = 'debug'
  log: bool = True
  log_dir: str = '/content/assignment/logs'
  seed: int = 42

  # Miscellaneous
  num_workers: int = 2
  device: str = 'cuda'
  progress_bar: bool = False
  print_every: int = 10


# The 6 configurations you need to run in Problem 1. Be careful that there is no discrepency between the configurations defined in `run_exp_lstm.py` and the ones below. In case there is a difference, the version from `run_exp_lstm.py` should be considered the ones to run.

# In[ ]:


# Note: if there is any discrepency with the configurations in run_exp_lstm.py, the
# version from run_exp_lstm.py should be the ones to use in Problem 1.
configs = {
  1: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='adam'),
  2: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='adamw'),
  3: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='sgd'),
  4: Arguments(model='lstm', layers=1, batch_size=16, log=True, epochs=10, optimizer='momentum'),

  5: Arguments(model='lstm', layers=2, batch_size=16, log=True, epochs=10, optimizer='adamw'),
  6: Arguments(model='lstm', layers=4, batch_size=16, log=True, epochs=10, optimizer='adamw')
}


# In[ ]:


def main(args):
  # Seed the experiment, for repeatability
  seed_experiment(args.seed)

  # Dataloaders
  train_dataset = Wikitext2(args.data_folder, split="train")
  train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
  )

  valid_dataset = Wikitext2(args.data_folder, split="validation")
  valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
  )

  test_dataset = Wikitext2(args.data_folder, split="test")
  test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
  )

  # Download the embeddings
  if not os.path.isfile(args.embeddings):
    print("Downloading embeddings...")
    urllib.request.urlretrieve(EMBEDDINGS_URL, args.embeddings)

  # Model
  if args.model == "lstm":
    model = LSTM.load_embeddings_from(
      args.embeddings, hidden_size=512, num_layers=args.layers
    )
  elif args.model == "gpt1":
    model = MiniGPT1.load_embeddings_from(
      args.embeddings, num_layers=args.layers
    )
  else:
    raise ValueError("Unknown model {0}".format(args.model))
  model.to(args.device)

  # Optimizer
  if args.optimizer == "adamw":
    optimizer = optim.AdamW(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
  elif args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
  elif args.optimizer == "sgd":
    optimizer = optim.SGD(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
  elif args.optimizer == "momentum":
    optimizer = optim.SGD(
      model.parameters(),
      lr=args.lr,
      momentum=args.momentum,
      weight_decay=args.weight_decay,
    )

  print(
    f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} "
    f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable."
  )

  train_losses, valid_losses = [], []
  train_ppls, valid_ppls = [], []
  train_times, valid_times = [], []
  for epoch in range(args.epochs):

    tqdm.write(f"====== Epoch {epoch} ======>")

    loss, ppl, wall_time = train(epoch, model, train_dataloader, optimizer, args)
    train_losses.append(loss)
    train_ppls.append(ppl)
    train_times.append(wall_time)

    loss, ppl, wall_time = evaluate(epoch, model, valid_dataloader, args)
    valid_losses.append(loss)
    valid_ppls.append(ppl)
    valid_times.append(wall_time)

  test_loss, test_ppl, test_time = evaluate(
    epoch, model, test_dataloader, args, mode="test"
  )

  print(f"===== Best validation perplexity: {min(valid_ppls):.3f} =====>")

  return (
    train_losses,
    train_ppls,
    train_times,
    valid_losses,
    valid_ppls,
    valid_times,
    test_loss,
    test_ppl,
    test_time,
  )


# In[ ]:


args = configs[1]  # Run the first configuration
logs = main(args)
if args.log:
  save_logs(args, *logs)


# Similarly you can add and run the configs listed in 'run_exps_vit.py' for the problem 3

# In[ ]:




