"""

Text Classification

 Data  : IMDB sentiment
 Model : LSTM RNN	

"""

from torchtext import data, datasets
from torchtext.vocab import GloVe

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import argparse
import os

# cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='train', help='run mode : train/evaluate/predict')
parser.add_argument("--hidden_dim", default=256, help='size of hidden state')
parser.add_argument("--batch_size", default=32, help='batch size for training')
parser.add_argument("--input",  
    #default='Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money.',
    default='This is one of the best creation of Nolan. I can say, it\'s his magnum opus. Loved the soundtrack and especially those creative dialogues.',
    help='input sentence to run prediction on'
    )
args, unknown = parser.parse_known_args()

# settings
MODEL_SAVE_PATH='.model'
MODEL_SAVE_FILE=os.path.join(MODEL_SAVE_PATH, 'lstm.pt')
# create dir if necessary
if not os.path.isdir(MODEL_SAVE_PATH):
  os.makedirs(MODEL_SAVE_PATH)


def load_data(batch_size=32):
  # define a tokenizer
  # tokenize = lambda s : nltk.word_tokenize(s)
  tokenize = lambda s : s.split()
  # fields : ( text_field, label_field )
  print(':: creating fields')
  text_field = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
  #text_field  = data.Field(sequential=True, tokenize=tokenize, lower=True)
  label_field = data.LabelField(sequential=False)
  # get IMDB data
  print(':: fetching IMDB data')
  train_data, test_data = datasets.IMDB.splits(text_field, label_field) 
  # build vocabulary for fields
  text_field.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
  label_field.build_vocab(train_data)

  # split train into train and valid
  train_data, valid_data = train_data.split() 

  print(':: labels :', label_field.vocab.stoi)

  # iterators
  train_iter, test_iter, valid_iter = data.BucketIterator.splits( 
                  (train_data, test_data, valid_data), 
                  batch_size=batch_size, 
                  sort_key=lambda x : len(x.text),
                  repeat=False,
                  shuffle=True)

  return  ( (text_field, label_field), (train_iter, test_iter, valid_iter), 
      text_field.vocab.vectors, # GloVe vectors 
      len(text_field.vocab)
        )
	
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


class LstmClassifier(nn.Module):

  def __init__(self, hparams, weights=None):
    """
    LSTM RNN Classifier

    Args:
      hparams : dictionary of hyperparameters
      
    """
    super(LstmClassifier, self).__init__()

    self.hparams = hparams
    self.weights = weights
    # init embedding lookup
    self.embedding = nn.Embedding(hparams['vocab_size'], hparams['emb_dim'])
    # set learned weights
    #  disable training
    if weights:
      self.embedding.weight = nn.Parameter(weights['glove'], requires_grad=False)
    # lstm 
    self.lstm = nn.LSTM(hparams['emb_dim'], hparams['hidden_dim'])
    # linear layer
    self.linear = nn.Linear(hparams['hidden_dim'], hparams['output_size'])

  def forward(self, sequence, batch_size=None, get_hidden=False):
    """
    Forward Operation.

    Args:
      sequence : list of indices based off a sentence

    """
    # infer batch_size and seqlen
    #print(sequence.size())
    # restructure sequence
    #sequence = sequence.permute(1, 0)
    # embed input
    input = self.embedding(sequence)
    input = input.permute(1, 0, 2)

    # initial state
    batch_size = batch_size if batch_size else self.hparams['batch_size']
    if torch.cuda.is_available():
      h0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']).cuda())
      c0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']).cuda())
    else:
      h0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']))
      c0 = Variable(torch.zeros(1, batch_size, self.hparams['hidden_dim']))
    
    # fix for "RNN weights not part of single contiguous chunk of memory" issue
    #  https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
    self.lstm.flatten_parameters()
    lstm_out, (h, c) = self.lstm(input, (h0, c0))

    # expose final state/representation
    self.h = h[-1]

    # linear layer
    linear_out = self.linear(h0[-1]) # NOTE BUG planted here
    #linear_out = self.linear(h[-1])

    # softmax layer
    # softmax_out = F.log_softmax(linear_out, dim=-1)
    
    if get_hidden:
      return linear_out, self.h

    return linear_out


def train_epoch(model, train_iter, hparams):
  # prepare model for training
  if torch.cuda.is_available():
    model.cuda()
  # train mode
  model.train()

  # loss function
  loss_fn = hparams['loss_fn']
  
  optim = torch.optim.Adam([ p for p in model.parameters() if p.requires_grad ])
  steps = 0
  epoch_loss, epoch_accuracy = 0, 0
  for idx, batch in enumerate(train_iter):
    # (1) clear gradients
    optim.zero_grad() # NOTE : why did I do model.zero_grad() ?

    # (2) inputs and targets
    inputs, targets = batch.text[0], batch.label
    # if cuda
    if torch.cuda.is_available():
      inputs = inputs.cuda()
      targets = targets.cuda()

    if inputs.size()[0] is not hparams['batch_size']:
      continue

    # (3) forward pass
    likelihood = model(inputs)

    # (4) loss calculation
    loss = loss_fn(likelihood, targets)
    # add to epoch loss
    epoch_loss += loss.item()

    # (5) optimization
    loss.backward()
    clip_gradient(model, 1e-1)
    optim.step()
    steps += 1
    epoch_loss += loss.item()

    num_corrects = (torch.max(likelihood, 1)[1].view(targets.size()).data == targets.data).float().sum()
    acc = 100.0 * num_corrects/len(batch)
    epoch_accuracy += acc

    if idx and idx%100 == 0:
      print('({}) Iteration loss : {}'.format(idx, loss.item()))
      
  print('Epoch loss : {}, Epoch accuracy : {}%'.format(epoch_loss/steps, epoch_accuracy/steps))

  return epoch_loss/steps, epoch_accuracy/steps

def evaluate(model, test_iter, hparams):
  epoch_loss, epoch_accuracy = 0., 0.
  loss_fn = hparams['loss_fn']

  # prepare model for evaluation
  model.eval()
  if torch.cuda.is_available():
    model.cuda()

  steps = 0
  with torch.no_grad():
    for idx, batch in enumerate(test_iter):

      # (1) get inputs and targets
      inputs, targets = batch.text[0], batch.label

      # if cuda
      if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

      if inputs.size()[0] is not 32:
        continue

      # (2) forward
      likelihood = model(inputs)

      # (3) loss calc
      loss = loss_fn(likelihood, targets)
      epoch_loss += loss.item()

      # (4) accuracy calc
      num_corrects = (torch.max(likelihood, 1)[1].view(targets.size()).data == targets.data).float().sum()
      acc = 100.0 * num_corrects/len(batch)
      epoch_accuracy += acc.item()

      steps += 1

    print('::[evaluation] Loss : {}, Accuracy : {}'.format(
      epoch_loss/(steps), epoch_accuracy/(steps)))

    return epoch_loss/steps, epoch_accuracy/steps
    
def training(model, hparams, train_iter, valid_iter, epochs=10):

  # NOTE select best parameters based on accuracy on validation set
  ev_accuracies = []
  for epoch in range(epochs):
    print('[{}]'.format(epoch+1))
    tr_loss, tr_accuracy = train_epoch(model, train_iter, hparams)
    ev_loss, ev_accuracy = evaluate(model, valid_iter, hparams)
    
    # check for best parameters criterion
    if len(ev_accuracies) and ev_accuracy > max(ev_accuracies):
      torch.save(model, MODEL_SAVE_FILE)

    # keep track of evaluation accuracy
    ev_accuracies.append(ev_loss)

def encode(example, _fields):
  text_field, label_field = _fields
  enc_text  = torch.LongTensor(
      [ text_field.vocab.stoi[token] for token in example.text ]
  ).view(1, -1).cuda()
  return enc_text, label_field.vocab.stoi[example.label]

def encode_label(example, _fields):
  text_field, label_field = _fields
  return label_field.vocab.stoi[example.label]

def predict(model, sentence, _fields):
  # expand fields
  text_field, label_field = _fields
  # encode sentence
  encoded_sequence = torch.LongTensor([ text_field.vocab.stoi[token]
      for token in text_field.preprocess(sentence) ]).view(1, -1)
  if torch.cuda.is_available():
    encoded_sequence = encoded_sequence.cuda()

  # forward; explicitly state batch_size
  with torch.no_grad():
    likelihood = model(encoded_sequence, batch_size=1)

  sentiment = label_field.vocab.itos[
      torch.softmax(likelihood.view(2), dim=-1).argmax().item()
      ]
  # present results
  print('\ninput : {}\noutput : {}\n'.format(sentence, sentiment))

  return sentiment

def load_model():

  # check if trained model exists
  if os.path.exists(MODEL_SAVE_FILE):
    return torch.load(MODEL_SAVE_FILE)

  # load data from IMDB dataset
  _fields, _iters, glove_emb, vocab_size = load_data(batch_size=args.batch_size)
  text_field, label_field = _fields # NOTE _<var> : something to be expanded
  train_iter, test_iter, valid_iter = _iters

  # define a loss function
  loss_fn = F.cross_entropy
  
  # set hyperparameters
  hparams = { 
    'vocab_size'  : vocab_size, 
    'emb_dim'     : glove_emb.size()[-1],
    'hidden_dim'  : args.hidden_dim,
    'lr'          : 2e-5,
    'output_size' : 2,
    'loss_fn'     : loss_fn,
    'batch_size'  : args.batch_size
    }

  # create LSTM model
  lstmClassifier = LstmClassifier( hparams,
      weights = { 'glove' : glove_emb }
      )

  # train model
  training(lstmClassifier, hparams, train_iter, valid_iter, epochs=10)

  return model


if __name__ == '__main__':

  # load data from IMDB dataset
  _fields, _iters, glove_emb, vocab_size = load_data(batch_size=args.batch_size)
  text_field, label_field = _fields # NOTE _<var> : something to be expanded
  train_iter, test_iter, valid_iter = _iters

  # define a loss function
  loss_fn = F.cross_entropy
  
  # set hyperparameters
  hparams = { 
    'vocab_size'  : vocab_size, 
    'emb_dim'     : glove_emb.size()[-1],
    'hidden_dim'  : args.hidden_dim,
    'lr'          : 2e-5,
    'output_size' : 2,
    'loss_fn'     : loss_fn,
    'batch_size'  : args.batch_size
    }

  if args.mode == 'predict':
    # load trained model from file
    model = torch.load(MODEL_SAVE_FILE)  
    predict(model, args.input, _fields)

  elif args.mode == 'train':
    # create LSTM model
    lstmClassifier = LstmClassifier( hparams,
        weights = { 'glove' : glove_emb }
        )
    # train model
    training(lstmClassifier, hparams, train_iter, valid_iter, epochs=10)

  elif args.mode == 'evaluate':
    # load trained model
    model = torch.load(MODEL_SAVE_FILE)
    # run evaluation
    ev_loss, ev_accuracy = evaluate(model, valid_iter, hparams)
