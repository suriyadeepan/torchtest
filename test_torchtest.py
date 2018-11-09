from torchtest.torchtest import test_suite
from torchtest.torchtest import assert_uses_gpu
from torchtest.torchtest import assert_vars_change
from torchtest.torchtest import assert_vars_same
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
  model = tc.LstmClassifier(hparams, weights={
    'glove' : torch.rand(hparams['vocab_size'], hparams['emb_dim'])
    }) 

  # create a random batch
  #  lets say seq_len = 15
  batch = [ 
      torch.randint(0, hparams['vocab_size'], (hparams['batch_size'], 15)).long(), 
      torch.randint(0, hparams['output_size'], (hparams['batch_size'],)).long() 
      ]
 
  # check if GPU is in use
  # assert_uses_gpu()

  # run all tests
  test_suite(
      model, hparams['loss_fn'],
      torch.optim.Adam([p for p in model.parameters() if p.requires_grad]), 
      batch,
      non_train_vars= [ # embedding is supposed to be fixed 
        ('embedding.weight', model.embedding.weight) # variable(s) to check for change
        ],
      test_gpu_available=True
      )

  # test for change in a subset of variables
  """
  assert_vars_same(
      model, hparams['loss_fn'],
      torch.optim.Adam([p for p in model.parameters() if p.requires_grad]), 
      batch, 
      [ ('embedding.weight', model.embedding.weight) ] # variable(s) to check for change
      )
  """
