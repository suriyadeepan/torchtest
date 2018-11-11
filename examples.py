import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchtest import torchtest as tt


"""
[1] Variables Change

"""
inputs = Variable(torch.randn(20, 20))
targets = Variable(torch.randint(0, 2, (20,))).long()
batch = [inputs, targets]
model = nn.Linear(20, 2)

# what are the variables?
print('Our list of parameters', [ np[0] for np in model.named_parameters() ])

# do they change after a training step?
#  let's run a train step and see
tt.assert_vars_change(
    model=model, 
    loss_fn=F.cross_entropy, 
    optim=torch.optim.Adam(model.parameters()),
    batch=batch)

# let's try to break this, so the test fails
params_to_train = [ np[1] for np in model.named_parameters() if np[0] is not 'bias' ]
# run test now
""" FAILURE
tt.assert_vars_change(
    model=model, 
    loss_fn=F.cross_entropy, 
    optim=torch.optim.Adam(params_to_train),
    batch=batch)
"""

# YES! bias did not change
# What if bias is not supposed to change, by design?

"""
[2] Variables Don't Change

"""
# test to see if bias remains the same after training
tt.assert_vars_same(
    model=model, 
    loss_fn=F.cross_entropy, 
    optim=torch.optim.Adam(params_to_train),
    batch=batch,
    params=[('bias', model.bias)] 
    )
# it does? good. let's move on

"""
[3] Output Range

"""
# we are keeping the bias fixed for a reason
optim = torch.optim.Adam(params_to_train)
loss_fn=F.cross_entropy

tt.test_suite(model, loss_fn, optim, batch, 
    output_range=(-2, 2),
    test_output_range=True
    )

# seems to work
#  let's tweak the model to fail the test
model.bias = nn.Parameter(2 + torch.randn(2, ))

"""FAILURE
tt.test_suite(
    model,
    loss_fn, optim, batch, 
    output_range=(-1, 1),
    test_output_range=True
    )
"""

# as expected, it fails; yay!

"""FAILURE
[4] NaN

model.bias = nn.Parameter(float('NaN') * torch.randn(2, ))

tt.test_suite(
    model,
    loss_fn, optim, batch, 
    test_nan_vals=True
    )

"""
# okay, then

""""FAILURE
[4] Inf

model.bias = nn.Parameter(float('Inf') * torch.randn(2, ))

tt.test_suite(
    model,
    loss_fn, optim, batch, 
    test_inf_vals=True
    )
"""
