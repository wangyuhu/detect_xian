import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from read_ImageNetData import ImageNetData

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNetLayer(nn.Module):
    def __init__(self, block, n_blocks, in_channels, out_channels, stride):
        super().__init__()

        self.modules = []

        self.modules.append(block(in_channels, out_channels, stride))

        for _ in range(n_blocks - 1):
            self.modules.append(block(out_channels, out_channels, 1))

        self.blocks = nn.Sequential(*self.modules)

    def forward(self, x):
        return self.blocks(x)


class ResNet18(nn.Module):
    def __init__(self, layer, block):
        super().__init__()

        n_blocks = [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = layer(block, n_blocks[0], 64, 64, 1)
        self.layer2 = layer(block, n_blocks[1], 64, 128, 2)
        self.layer3 = layer(block, n_blocks[2], 128, 256, 2)
        self.layer4 = layer(block, n_blocks[3], 256, 512, 2)
        self.fc = nn.Linear(512*49, 54)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, layer, block):
        super().__init__()

        n_blocks = [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = layer(block, n_blocks[0], 64, 64, 1)
        self.layer2 = layer(block, n_blocks[1], 64, 128, 2)
        self.layer3 = layer(block, n_blocks[2], 128, 256, 2)
        self.layer4 = layer(block, n_blocks[3], 256, 512, 2)
        self.fc = nn.Linear(512*49, 54)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        fx = model(x)

        loss = criterion(fx, y)

        acc = calculate_accuracy(fx, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            fx = model(x)

            loss = criterion(fx, y)

            acc = calculate_accuracy(fx, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_transforms = transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(10),
                           transforms.RandomCrop(32, padding=3),
                           transforms.ToTensor(),
                           transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
                       ])

# train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transforms)
# test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transforms)
#
# n_train_examples = int(len(train_data)*0.9)
# n_valid_examples = len(train_data) - n_train_examples
#
# train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])

BATCH_SIZE = 16

dataloders, dataset_sizes = ImageNetData(data_dir="ImageData",batch_size=BATCH_SIZE,num_workers=0)
train_iterator = dataloders['train']
valid_iterator = dataloders['val']
# train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
# valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
# test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)


device = torch.device('cuda')
model = ResNet34(ResNetLayer, ResNetBlock).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

EPOCHS = 150
SAVE_DIR = 'models/resnet34'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet-xian.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')
val_result = {'epoch':[],'loss':[],'accy':[]}
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, device, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, device, valid_iterator, criterion)
    val_result['epoch'].append(epoch)
    val_result['loss'].append(valid_loss)
    val_result['accy'].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    plt.clf()
    plt.plot(val_result['epoch'], val_result['loss'])
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    plt.title('loss')
    plt.savefig(os.path.join(SAVE_DIR, 'loss.png'))

    plt.clf()
    plt.plot(val_result['epoch'], val_result['accy'])
    plt.xlabel('epoch')
    plt.ylabel('val_accy')
    plt.title('accy')
    plt.savefig(os.path.join(SAVE_DIR, 'accy.png'))
    print(
        f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:05.2f}% |')
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss, test_acc = evaluate(model, device, valid_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:05.2f}% |')


