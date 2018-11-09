from torchtest.torchtest import test_suite
import tc

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


if __name__ == '__main__':
  # define model params
  hparams={
    'vocab_size'  : 100, 
    'emb_dim'     : 20,
    'hidden_dim'  : 30,
    'output_size' : 2,
    'loss_fn'     : F.cross_entropy,
    'batch_size'  : 10
    }

  # create model
  model = tc.LstmClassifier(hparams) 

  # create a random batch
  #  lets say seq_len = 15
  batch = [ 
      torch.randint(0, hparams['vocab_size'], (hparams['batch_size'], 15)).long(), 
      torch.randint(0, hparams['output_size'], (hparams['batch_size'],)).long() 
      ]
 
  # run all tests
  test_suite(
      model, hparams['loss_fn'],
      torch.optim.Adam([p for p in model.parameters() if p.requires_grad]), 
      batch
      )
