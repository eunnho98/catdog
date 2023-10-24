from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader

class Trainer():

  def __init__(self, model, optimizer, crit, device):
    self.model = model
    self.optimizer = optimizer
    self.crit = crit
    self.device = device
  def _train(self, dataset, config):
    self.model.train() # train() or eval()

    train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)

    total_loss = 0

    for idx, samples in enumerate(train_loader):
      X_train, y_train = samples
      X_train = X_train.to(self.device)
      y_train = y_train.to(self.device)
      print('X_train size:', X_train.size())
      print('y_train size:', y_train.size())
      y_hat_i = self.model(X_train)
      print('y_hat', y_hat_i.squeeze())
      print('y_train', y_train)
      loss = self.crit(y_hat_i.squeeze(), y_train.float())

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if config.verbose >= 2:
        print('Train iteration(%d%d): loss=%.4e' % (idx+1, len(train_loader), float(loss)))

      total_loss += float(loss)
    return total_loss / len(train_loader)
  
  def _validate(self, dataset, config):
    self.model.eval()

    with torch.no_grad():
      valid_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False)
      total_loss = 0

      for idx, samples in enumerate(valid_loader):
        X_valid, y_valid = samples
        X_valid = X_valid.to(self.device)
        y_valid = y_valid.to(self.device)
        y_hat_i = self.model(X_valid)
        loss = self.crit(y_hat_i, y_valid.squeeze())

        if config.verbose >= 2:
          print('Valid Iteration(%d%d): loss=%.4e' (idx + 1, len(valid_loader), float(loss)))

        total_loss = float(loss)

      return total_loss / len(valid_loader)
    
  # 매 에포크마다 Train
  def train(self, train_data, valid_data, config):
    lowest_loss = np.inf
    best_model = None

    for epoch_index in range(config.n_epochs):
      train_loss = self._train(train_data, config)
      valid_loss = self._validate(valid_data, config)

      if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        best_model = deepcopy(self.model.state_dict())

      print('Epoch(%d%d): train_loss=%.4e, valid_loss=%.4e, lowest_loss=%.4e' % (epoch_index + 1, config.n_epochs, train_loss, valid_loss, lowest_loss))

    self.model.load_state_dict(best_model)




