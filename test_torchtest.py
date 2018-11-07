from torchtest.torchtest import _var_change_helper
import tc

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


if __name__ == '__main__':

  # create model
  model = tc.LstmClassifier({ 
    'vocab_size'  : 100, 
    'emb_dim'     : 20,
    'hidden_dim'  : 30,
    'output_size' : 2,
    'loss_fn'     : F.cross_entropy,
    'batch_size'  : 10
    })

  # create a random batch
  batch = [ torch.randint(0, 100, (10, 15)).long(), torch.randint(0, 2, (10,)).long() ]
 
  # test variable change
  _var_change_helper(True, model, F.cross_entropy, 
      torch.optim.Adam([p for p in model.parameters() if p.requires_grad]), 
      batch)
