"""torchtest : A Tiny Test Suite for PyTorch

A tiny test suite for pytorch based Machine Learning models, inspired by mltest. 

Chase Roberts lists out 4 basic tests in his medium post about mltest. 
https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d

torchtest is sort of a pytorch port of mltest (which was written for tensorflow models).
"""

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
  """Set random seed for torch"""
  torch.manual_seed(seed)

def _train_step(model, loss_fn, optim, batch, device):
  """Run a training step on model for a given batch of data

  Parameters of the model accumulate gradients and the optimizer performs
  a gradient update on the parameters

  Parameters
  ----------
  model : torch.nn.Module
    torch model, an instance of torch.nn.Module
  loss_fn : function
    a loss function from torch.nn.functional 
  optim : torch.optim.Optimizer
    an optimizer instance
  batch : list
    a 2 element list of inputs and labels, to be fed to the model
  """

  # put model in train mode
  model.train()
  model.to(device)

  #  run one forward + backward step
  # clear gradient
  optim.zero_grad()
  # inputs and targets
  inputs, targets = batch[0], batch[1]
  # move data to DEVICE
  inputs = inputs.to(device)
  targets = targets.to(device)
  # forward
  likelihood = model(inputs)
  # calc loss
  loss = loss_fn(likelihood, targets)
  # backward
  loss.backward()
  # optimization step
  optim.step()

def _forward_step(model, batch, device):
  """Run a forward step of model for a given batch of data

  Parameters
  ----------
  model : torch.nn.Module
    torch model, an instance of torch.nn.Module
  batch : list
    a 2 element list of inputs and labels, to be fed to the model

  Returns
  -------
  torch.tensor
    output of model's forward function 
  """

  # put model in eval mode
  model.eval()
  model.to(device)

  with torch.no_grad():
    # inputs and targets
    inputs = batch[0]
    # move data to DEVICE
    inputs = inputs.to(device)
    # forward
    return model(inputs)

def _var_change_helper(vars_change, model, loss_fn, optim, batch, device, params=None): 
  """Check if given variables (params) change or not during training

  If parameters (params) aren't provided, check all parameters.

  Parameters
  ----------
  vars_change : bool
    a flag which controls the check for change or not change
  model : torch.nn.Module
    torch model, an instance of torch.nn.Module
  loss_fn : function
    a loss function from torch.nn.functional 
  optim : torch.optim.Optimizer
    an optimizer instance
  batch : list
    a 2 element list of inputs and labels, to be fed to the model
  params : list, optional
    list of parameters of form (name, variable)

  Raises
  ------
  VariablesChangeException
    if vars_change is True and params DO NOT change during training
    if vars_change is False and params DO change during training
  """

  if params is None:
    # get a list of params that are allowed to change
    params = [ np for np in model.named_parameters() if np[1].requires_grad ]

  # take a copy
  initial_params = [ (name, p.clone()) for (name, p) in params ]

  # run a training step
  _train_step(model, loss_fn, optim, batch, device)

  # check if variables have changed
  for (_, p0), (name, p1) in zip(initial_params, params):
    try:
      if vars_change:
        assert not torch.equal(p0.to(device), p1.to(device))
      else:
        assert torch.equal(p0.to(device), p1.to(device))
    except AssertionError:
      raise VariablesChangeException( # error message
          "{var_name} {msg}".format(
            var_name=name, 
            msg='did not change!' if vars_change else 'changed!' 
            )
          )

def assert_uses_gpu():
  """Make sure GPU is available and accessible

  Raises
  ------
  GpuUnusedException
    If GPU is inaccessible
  """

  try:
    assert torch.cuda.is_available()
  except AssertionError:
    raise GpuUnusedException(
        "GPU inaccessible"
        )

def assert_vars_change(model, loss_fn, optim, batch, device, params=None):
  """Make sure that the given parameters (params) DO change during training

  If parameters (params) aren't provided, check all parameters.

  Parameters
  ----------
  model : torch.nn.Module
    torch model, an instance of torch.nn.Module
  loss_fn : function
    a loss function from torch.nn.functional 
  optim : torch.optim.Optimizer
    an optimizer instance
  batch : list
    a 2 element list of inputs and labels, to be fed to the model
  params : list, optional
    list of parameters of form (name, variable)

  Raises
  ------
  VariablesChangeException
    If params do not change during training
  """

  _var_change_helper(True, model, loss_fn, optim, batch, device, params)

def assert_vars_same(model, loss_fn, optim, batch, device, params=None):
  """Make sure that the given parameters (params) DO NOT change during training

  If parameters (params) aren't provided, check all parameters.

  Parameters
  ----------
  model : torch.nn.Module
    torch model, an instance of torch.nn.Module
  loss_fn : function
    a loss function from torch.nn.functional 
  optim : torch.optim.Optimizer
    an optimizer instance
  batch : list
    a 2 element list of inputs and labels, to be fed to the model
  params : list, optional
    list of parameters of form (name, variable)

  Raises
  ------
  VariablesChangeException
    If params change during training
  """

  _var_change_helper(False, model, loss_fn, optim, batch, device, params)

def assert_any_greater_than(tensor, value):
  """Make sure that one or more elements of tensor greater than value

  Parameters
  ----------
  tensor : torch.tensor
    input tensor 
  value : float
    numerical value to check against

  Raises
  ------
  RangeException
    If all elements of tensor are less than value 
  """

  try:
    assert (tensor > value).byte().any()
  except AssertionError:
    raise RangeException(
        "All elements of tensor are less than {value}".format(
          value=value)
        )

def assert_all_greater_than(tensor, value):
  """Make sure that all elements of tensor are greater than value

  Parameters
  ----------
  tensor : torch.tensor
    input tensor 
  value : float
    numerical value to check against

  Raises
  ------
  RangeException
    If one or more elements of tensor are less than value 
  """

  try:
    assert (tensor > value).byte().all()
  except AssertionError:
    raise RangeException(
        "Some elements of tensor are less than {value}".format(
          value=value)
        )

def assert_any_less_than(tensor, value):
  """Make sure that one or more elements of tensor are less than value

  Parameters
  ----------
  tensor : torch.tensor
    input tensor 
  value : float
    numerical value to check against

  Raises
  ------
  RangeException
    If all elements of tensor are greater than value 
  """

  try:
    assert (tensor < value).byte().any()
  except AssertionError:
    raise RangeException(
        "All elements of tensor are greater than {value}".format(
          value=value)
        )

def assert_all_less_than(tensor, value):
  """Make sure that all elements of tensor are less than value

  Parameters
  ----------
  tensor : torch.tensor
    input tensor 
  value : float
    numerical value to check against

  Raises
  ------
  RangeException
    If one or more elements of tensor are greater than value 
  """

  try:
    assert (tensor < value).byte().all()
  except AssertionError:
    raise RangeException(
        "Some elements of tensor are greater than {value}".format(
          value=value)
        )

def assert_input_dependency(model, loss_fn, optim, batch, 
    independent_vars=None,
    dependent_vars=None):
  """Makes sure the "dependent_vars" are dependent on "independent_vars" """
  raise NotImplementedError("""
    I don't know a clean way to do this
    Doesn't assert_vars_change() cover this?
  """
  )


def assert_never_nan(tensor):
  """Make sure there are no NaN values in the given tensor.

  Parameters
  ----------
  tensor : torch.tensor
    input tensor 

  Raises
  ------
  NaNTensorException
    If one or more NaN values occur in the given tensor
  """

  try:
    assert not torch.isnan(tensor).byte().any()
  except AssertionError:
    raise NaNTensorException("There was a NaN value in tensor")

def assert_never_inf(tensor):
  """Make sure there are no Inf values in the given tensor.

  Parameters
  ----------
  tensor : torch.tensor
    input tensor 

  Raises
  ------
  InfTensorException
    If one or more Inf values occur in the given tensor
  """

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
    test_gpu_available=False,
    device='cpu'):
  """Test Suite : Runs the tests enabled by the user

  If output_range is None, output of model is tested against (MODEL_OUT_LOW, 
  MODEL_OUT_HIGH). 

  Parameters
  ----------
  model : torch.nn.Module
    torch model, an instance of torch.nn.Module
  loss_fn : function
    a loss function from torch.nn.functional 
  optim : torch.optim.Optimizer
    an optimizer instance
  batch : list
    a 2 element list of inputs and labels, to be fed to the model
  output_range : tuple, optional
    (low, high) tuple to check against the range of logits (default is
    None)
  train_vars : list, optional
    list of parameters of form (name, variable) to check if they change
    during training (default is None)
  non_train_vars : list, optioal
    list of parameters of form (name, variable) to check if they DO NOT
    change during training (default is None)
  test_output_range : boolean, optional
    switch to turn on or off range test (default is False)
  test_vars_change : boolean, optional
    switch to turn on or off variables change test (default is False)
  test_nan_vals : boolean, optional
    switch to turn on or off test for presence of NaN values (default is False)
  test_inf_vals : boolean, optional
    switch to turn on or off test for presence of Inf values (default is False)
  test_gpu_available : boolean, optional
    switch to turn on or off GPU availability test (default is False)

  Raises
  ------
  VariablesChangeException
    If selected params change/do not change during training
  RangeException
    If range of output exceeds the given limit
  GpuUnusedException
    If GPU is inaccessible
  NaNTensorException
    If one or more NaN values occur in model output
  InfTensorException
    If one or more Inf values occur in model output
  """

  # check if all variables change
  if test_vars_change:
    assert_vars_change(model, loss_fn, optim, batch, device)

  # check if train_vars change
  if train_vars is not None:
    assert_vars_change(model, loss_fn, optim, batch, device, params=train_vars)

  # check if non_train_vars don't change
  if non_train_vars is not None:
    assert_vars_same(model, loss_fn, optim, batch, device, params=non_train_vars)

  # run forward once
  model_out = _forward_step(model, batch, device)

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
