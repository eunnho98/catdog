import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tr

from utils import load_img
from utils import split_data
from utils import get_model
from Trainer import Trainer

def define_argparser():
  p = argparse.ArgumentParser(description='Save input parameters to config Object')

  p.add_argument('--model_fn', required=True, help='File path where model weights will be saved')
  p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1, help='GPU idx number, default = 0, if not available -1')
  
  p.add_argument('--train_ratio', type=float, default=.8, help='Train data ratio, default=0.8')
  p.add_argument('--batch_size', type=int, default=256, help='mini_batch size, default=256')
  p.add_argument('--n_epochs', type=int, default=20, help='n_epochs, default=20')

  p.add_argument('--n_layers', type=int, default=5, help='Number of Model Layers, default=5')
  p.add_argument('--use_dropout', action='store_true', help='if input --use_dropout, then you use dropout')
  p.add_argument('--dropout_p', type=float, default=.3, help='dropout probablility, default=0.3')

  p.add_argument('--verbose', type=int, default=1, help='Degree of log output during training')

  p.add_argument('--model', default='cnn', choices=['fc', 'cnn'], help='Choose Model')

  config = p.parse_args()

  return config

def main(config):
  device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
  train_transform = tr.Compose([
    tr.Resize((256, 256)),
    tr.RandomCrop(224),
    tr.RandomHorizontalFlip(),
    tr.ToTensor()
  ])

  test_transform = tr.Compose([
    tr.Resize((224, 224)),
    tr.ToTensor()
  ])

  train = load_img(train_transform=train_transform, is_train=True)

  train_dataset, valid_dataset = split_data(train, train_ratio=.8) # X[0]: train_dataset, X[1]: valid_dataset

  input_size = int(train[0][0].size(-1))
  output_size = 1

  channel_num = 1
  if len(train[0][0].size()) == 3:
    channel_num = 3
  

  model = get_model(
    input_size,
    output_size,
    config,
    channel_num,
    device
  ).to(device)
  optimizer = optim.Adam(model.parameters())
  crit = nn.BCELoss()

  if config.verbose >= 1:
    print(model)
    print(optimizer)
    print(crit)

  trainer = Trainer(model, optimizer, crit, device)

  trainer.train(
    train_data=train_dataset,
    valid_data=valid_dataset,
    config=config
  )

  torch.save({
    'model': trainer.model.state_dict(),
    'opt': optimizer.state_dict(),
    'config': config
  }, config.model_fn)
  


if __name__ == '__main__':
  config = define_argparser()
  main(config)