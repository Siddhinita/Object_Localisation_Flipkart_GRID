import os
import torch
import torchvision
from main import FlipkartDataset, ResNet18Model
import torch.nn as nn
from torch.utils.data import DataLoader

cuda = torch.cuda.is_available()

model = ResNet18Model()

if cuda:
    model = model.cuda()
if os.path.exists('resnet18_baseline.pth.tar'):
    best_model_weights = torch.load('resnet18_baseline.pth.tar')
    model.load_state_dict(best_model_weights)
else:
	print('weights not found!')

model.eval()

dataset = FlipkartDataset('test2.csv', 'images/')
datagen = DataLoader(dataset, batch_size=32)

df = dataset.data_df
outs = []

x = 0

for samples in datagen:
    images = samples['image']
    if cuda:
        images = images.cuda()
    o = model(images).cpu().data
    outs.append(o)
    x += images.size(0)
    print('\r{:5d}'.format(x), end='')
print()

outs = torch.cat(outs).cpu().numpy()

df['x1'] = (outs[:, 0] - outs[:, 2] * 0.5) * 640
df['y1'] = (outs[:, 1] - outs[:, 3] * 0.5) * 480
df['x2'] = (outs[:, 0] + outs[:, 2] * 0.5) * 640
df['y2'] = (outs[:, 1] + outs[:, 3] * 0.5) * 480

df.to_csv('submission.csv', index=False, float_format='%.3f')
