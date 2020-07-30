import os
import cv2
import torch
import pandas as pd
import numpy as np
from time import time
from skimage import io, transform
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.autograd import Variable


def normalize(x1, y1, x2, y2, W=640., H=480.):
    dw = 1. / W
    dh = 1. / H
    xc = dw * (x1 + x2) * 0.5
    yc = dh * (y1 + y2) * 0.5
    w = dw * (x2 - x1)
    h = dh * (y2 - y1)
    return xc, yc, w, h


def denormalize(xc, yc, w, h, W=640., H=480.):
    x1 = W * (xc - w * 0.5)
    y1 = H * (yc - h * 0.5)
    x2 = W * (xc + w * 0.5)
    y2 = H * (yc + h * 0.5)
    return x1, y1, x2, y2


class FlipkartDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        img_name = os.path.join(self.root_dir, row['image_name'])
        image = io.imread(img_name).transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        x1, x2 = row.iloc[1:3]
        y1, y2 = row.iloc[3:5]
        box = np.array(normalize(x1, y1, x2, y2))
        box = torch.from_numpy(box).float()

        sample = {'image': image, 'box': box}

        return sample


class ResNet18Model(nn.Module):
    def __init__(self):
        super(ResNet18Model, self).__init__()
        self.resnet18_top = torchvision.models.resnet18(pretrained=False)

        num_features = self.resnet18_top.fc.in_features
        self.resnet18_top.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet18_top.fc = nn.Linear(num_features, 4)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        out = self.resnet18_top(x)
        return self.sigmoid_layer(out)


def huber_loss(y_true, y_pred, huber=0.5):
    x = torch.abs(y_true - y_pred)
    x = torch.where(x < huber, 0.5 * x.pow(2), huber * (x - 0.5 * huber))
    return x.sum()


def bbox_iou(y_true, y_pred):
    b1_x1, b1_y1, b1_x2, b1_y2 = denormalize(y_true[:, 0], y_true[:, 1],
                                             y_true[:, 2], y_true[:, 3])
    b2_x1, b2_y1, b2_x2, b2_y2 = denormalize(y_pred[:, 0], y_pred[:, 1],
                                             y_pred[:, 2], y_pred[:, 3])

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou.sum()


def iou_loss(y_true, y_pred):
    b1_x1, b1_y1, b1_x2, b1_y2 = denormalize(y_true[:, 0], y_true[:, 1],
                                             y_true[:, 2], y_true[:, 3])
    b2_x1, b2_y1, b2_x2, b2_y2 = denormalize(y_pred[:, 0], y_pred[:, 1],
                                             y_pred[:, 2], y_pred[:, 3])

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    iou_loss = -torch.log(iou)

    return iou_loss.sum()


def train_model(model,
                cuda,
                dataloaders,
                optimizer,
                scheduler,
                dataset_sizes,
                num_epochs=25):
    since = time()
    best_model_weights = model.state_dict()
    best_iou = 0.0

    for epoch in range(num_epochs):

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_iou = 0.0
            x = 0

            for samples in dataloaders[phase]:

                images = samples['image']
                boxes = samples['box']

                if cuda:
                    images = images.cuda(async=True)
                    boxes = boxes.cuda(async=True)

                optimizer.zero_grad()

                outputs = model(images)
                loss = huber_loss(boxes, outputs)
                iou = bbox_iou(boxes, outputs)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_iou += iou

                print(
                    '\r{} Epoch {:2d}/{:2d} {:3d}%  {:5d}/{:5d} Loss {:.4f} IOU {:.4f}'
                    .format(phase, epoch + 1, num_epochs,
                            int(100 * x / dataset_sizes[phase]),
                            (x + images.size(0)), dataset_sizes[phase],
                            running_loss / (x + images.size(0)),
                            running_iou / (x + images.size(0))),
                    end='')

                x += images.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_iou = running_iou / dataset_sizes[phase]

            if phase == 'valid' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_weights = model.state_dict()
                torch.save(best_model_weights, 'resnet18_baseline.pth.tar')

            print()

    print('\n-' * 10)
    print()

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best IOU: {:4f}'.format(best_iou))

    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':

    root_dir = '/input/'
    batch_size = 32
    csv_files = {'train': 'TRAIN.csv', 'valid': 'VALIDATE.csv'}
    datasets = {x: FlipkartDataset(csv_files[x], root_dir) for x in csv_files}
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True)
        for x in datasets
    }
    dataset_sizes = {x: len(datasets[x]) for x in datasets}
    cuda = torch.cuda.is_available()

    model = ResNet18Model()

    if cuda:
        model = model.cuda()



    if os.path.exists('resnet18_baseline.pth.tar'):
        print('Resuming with pre-trained weights')
        #best_model_weights = torch.load('resnet18_baseline.pth.tar')
        #model.load_state_dict(best_model_weights)

    for param in model.resnet18_top.parameters():
        param.requires_grad = False
    for param in model.resnet18_top.fc.parameters():
        param.requires_grad = True

    print('Following parameters are trainable\n')
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            trainable_params.append(param)
    print()

    optimizer = optim.SGD(trainable_params, lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)

    model = train_model(model, cuda, dataloaders, optimizer, exp_lr_scheduler,
                        dataset_sizes)
