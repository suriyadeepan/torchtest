import torch


"""

Are we using the GPU?

"""
def test_uses_gpu():
  return torch.cuda.is_available()

def _var_change_helper(vars_change, model, loss_fn, optim, batch):
  # trainable parameters
  trainable_params = [ p for p in model.parameters() if p.requires_grad ]
  # take a copy
  initial_params = [ p.clone() for p in trainable_params ]
  # create optimizer
  optim = optim(trainable_params)
  # put model in train mode
  model.train()
  if torch.cuda.is_available():
    model.cuda()

  #  run one forward + backward step
  # clear gradient
  optim.zero_grad()
  # inputs and targets
  inputs, targets = batch[0], batch[1]
  # move data to GPU
  if torch.cuda.is_available():
    inputs = inputs.cuda()
    targets = targets.cuda()
  # forward
  likelihood = model(inputs)
  # calc loss
  loss = loss_fn(likelihood, targets)
  # backward
  loss.backward()
  # optimization step
  optim.step()

  # check if variables have changed
  for p0, p1 in zip(initial_params, 
      [ p for p in model.parameters() if p.requires_grad ]):
    if vars_change:
      assert not torch.equal(p0, p1)
    else:
      assert torch.equal(p0, p1)

def assert_any_greater_than(tensor, value):
  assert (tensor > value).byte().any()

def assert_all_greater_than(tensor, value):
  assert (tensor > value).byte().all()
  
def assert_any_less_than(tensor, value):
  assert (tensor < value).byte().any()

def assert_all_less_than(tensor, value):
  assert (tensor < value).byte().all()
