import torch

# default model output range
MODEL_OUT_LOW = -1
MODEL_OUT_HIGH = 1

class GpuUnusedException(Exception):
  pass

class VariablesChangeException(Exception):
  pass

class RangeException(Exception):
  pass

class DependencyException(Exception):
  pass

class NaNTensorException(Exception):
  pass

class InfTensorException(Exception):
  pass

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

def _forward_step(model, batch):
  # put model in eval mode
  model.eval()
  if torch.cuda.is_available():
    model.cuda()

  with torch.no_grad():
    # inputs and targets
    inputs = batch[0]
    # move data to GPU
    if torch.cuda.is_available():
      inputs = inputs.cuda()
    # forward
    return model(inputs)

def _var_change_helper(vars_change, model, loss_fn, optim, batch, params=None): 
  if params is None:
    # get a list of params that are allowed to change
    params = [ np for np in model.named_parameters() if np[1].requires_grad ]

  # take a copy
  initial_params = [ (name, p.clone()) for (name, p) in params ]

  # run a training step
  _train_step(model, loss_fn, optim, batch)

  # check if variables have changed
  for (_, p0), (name, p1) in zip(initial_params, params):
    try:
      if vars_change:
        assert not torch.equal(p0, p1)
      else:
        assert torch.equal(p0, p1)
    except AssertionError:
      raise VariablesChangeException( # error message
          "{var_name} {msg}".format(
            var_name=name, 
            msg='did not change!' if vars_change else 'changed!' 
            )
          )

def assert_uses_gpu():
  try:
    assert torch.cuda.is_available()
  except AssertionError:
    raise GpuUnusedException(
        "GPU inaccessible"
        )

def assert_vars_change(model, loss_fn, optim, batch, params=None):
  _var_change_helper(True, model, loss_fn, optim, batch, params)

def assert_vars_same(model, loss_fn, optim, batch, params=None):
  _var_change_helper(False, model, loss_fn, optim, batch, params)

def assert_any_greater_than(tensor, value):
  try:
    assert (tensor > value).byte().any()
  except AssertionError:
    raise RangeException(
        "All elements of tensor are less than {value}".format(
          value=value)
        )

def assert_all_greater_than(tensor, value):
  try:
    assert (tensor > value).byte().all()
  except AssertionError:
    raise RangeException(
        "Some elements of tensor are less than {value}".format(
          value=value)
        )

def assert_any_less_than(tensor, value):
  try:
    assert (tensor < value).byte().any()
  except AssertionError:
    raise RangeException(
        "All elements of tensor are greater than {value}".format(
          value=value)
        )

def assert_all_less_than(tensor, value):
  try:
    assert (tensor < value).byte().all()
  except AssertionError:
    raise RangeException(
        "Some elements of tensor are greater than {value}".format(
          value=value)
        )

def assert_input_dependency(model, loss_fn, optim, batch):
  # NOTE i don't know a clean way to do this
  #      doesn't assert_vars_change() cover this?
  pass 

def assert_never_nan(tensor):
  try:
    assert not torch.isnan(tensor).byte().any()
  except AssertionError:
    raise NaNTensorException("There was a NaN value in tensor")

def assert_never_inf(tensor):
  try:
    assert torch.isfinite(tensor).byte().any()
  except AssertionError:
    raise InfTensorException("There was an Inf value in tensor")

def test_suite(model, loss_fn, optim, batch,
    output_range=None,
    train_vars=None,
    non_train_vars=None,
    test_output_range=False,
    test_vars_change=False,
    test_nan_vals=False,
    test_inf_vals=False,
    test_gpu_available=False):

  # check if all variables change
  if test_vars_change:
    assert_vars_change(model, loss_fn, optim, batch)

  # check if train_vars change
  if train_vars is not None:
    assert_vars_change(model, loss_fn, optim, batch, params=train_vars)

  # check if non_train_vars don't change
  if non_train_vars is not None:
    assert_vars_same(model, loss_fn, optim, batch, params=non_train_vars)

  # run forward once
  model_out = _forward_step(model, batch)

  # range tests
  if test_output_range:
    if output_range is None:
      assert_all_greater_than(model_out, MODEL_OUT_LOW)
      assert_all_less_than(model_out, MODEL_OUT_HIGH)
    else:
      assert_all_greater_than(model_out, output_range[0])
      assert_all_less_than(model_out, output_range[1])

  # NaN Test
  if test_nan_vals:
    assert_never_nan(model_out)

  # Inf Test
  if test_inf_vals:
    assert_never_inf(model_out)

  # GPU test
  if test_gpu_available:
    assert_uses_gpu()
