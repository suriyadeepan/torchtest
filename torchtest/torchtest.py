import torch


def assert_uses_gpu():
  return torch.cuda.is_available()

def setup(seed=0):
  torch.manual_seed(seed)

def _train_step(model, loss_fn, optim, batch):
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

def _var_change_helper(vars_change, model, loss_fn, optim, batch):

  trainable_params = [ p for p in model.parameters() if p.requires_grad ]
  # take a copy
  initial_params = [ p.clone() for p in trainable_params ]

  # run a train step
  _train_step(model, loss_fn, optim, batch)

  # check if variables have changed
  for p0, p1 in zip(initial_params, 
      [ p for p in model.parameters() if p.requires_grad ]):
    if vars_change:
      assert not torch.equal(p0, p1)
    else:
      assert torch.equal(p0, p1)

def assert_vars_change(model, loss_fn, optim, batch):
  _var_change_helper(True, model, loss_fn, optim, batch)

def assert_vars_same(model, loss_fn, optim, batch):
  _var_change_helper(False, model, loss_fn, optim, batch)

def assert_any_greater_than(tensor, value):
  assert (tensor > value).byte().any()

def assert_all_greater_than(tensor, value):
  assert (tensor > value).byte().all()
  
def assert_any_less_than(tensor, value):
  assert (tensor < value).byte().any()

def assert_all_less_than(tensor, value):
  assert (tensor < value).byte().all()

def assert_input_dependency(model, loss_fn, optim, batch):
  pass

def assert_never_nan(tensor):
  assert not torch.isnan(tensor).byte().any()

def assert_never_inf(tensor):
  assert torch.isfinite(tensor).byte().any()
