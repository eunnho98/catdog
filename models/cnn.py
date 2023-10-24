import torch.nn as nn

# CNN Block
class ConvolutionalBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    self.in_channels = in_channels
    self.out_channels = out_channels

    super().__init__()

    self.layers = nn.Sequential(
      # (입력 채널 개수, 출력 채널 개수(커널개수), 커널 크기, 패딩 크기)
      # 입력 이미지의 특성을 감지하고 필터링
      nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(out_channels),
      # 생성된 특성 맵을 입력으로 받아 더 높은 수준의 특징을 추출
      nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1), # stride로 차원 축소
      nn.ReLU(),
      nn.BatchNorm2d(out_channels)
    )

  def forward(self, x):
    # x: (batch_size, in_channels, h, w)
    y = self.layers(x)
    # y: (batch_size, out_channels, h, w)
    return y
  
class CNNClassifier(nn.Module):

  def __init__(self, channel_num, output_size):
    self.channel_num = channel_num
    self.output_size = output_size

    super().__init__()
                                # batch_size = 256
    self.blocks = nn.Sequential( # x = (n, 3, 224, 224)
      ConvolutionalBlock(channel_num, 32), # x = (n, 32, 112, 112)
      ConvolutionalBlock(32, 64), # x = (n, 64, 56, 56)
      ConvolutionalBlock(64, 128), # x = (n, 128, 28, 28)
      ConvolutionalBlock(128, 256), # x = (n, 256, 14, 14)
      ConvolutionalBlock(256, 512) # x = (n, 512, 7, 7)
    )

    self.layers = nn.Sequential(
      nn.Linear(512 * 7 * 7, 256),
      nn.ReLU(),
      nn.BatchNorm1d(256),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.BatchNorm1d(128),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.BatchNorm1d(64),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.BatchNorm1d(32),
      nn.Linear(32, output_size),
      nn.Sigmoid()
    )

  def forward(self, x):
    assert x.dim() > 2, 'You ned to input matrix'

    if x.dim() == 3: # 흑백
      x = x.view(-1, self.channel_num, x.size(-2), x.size(-1)) # batch_size, 1, h, w

    z = self.blocks(x)
    z = z.view(z.size(0), -1) # (n) x (512*7*7)로 만들기 위해
    y = self.layers(z.squeeze()) # ((n) X *(512*7*7)) X ((512*7*7) X output_size)
    return y