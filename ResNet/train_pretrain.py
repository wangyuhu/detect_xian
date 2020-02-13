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
import torchvision.models as models
import matplotlib.pyplot as plt

from read_ImageNetData import ImageNetData


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
                           transforms.RandomCrop((224, 224), pad_if_needed=True),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

test_transforms = transforms.Compose([
                           transforms.CenterCrop((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

# train_data = datasets.ImageFolder('data/dogs-vs-cats/train', train_transforms)
# valid_data = datasets.ImageFolder('data/dogs-vs-cats/valid', test_transforms)
# test_data = datasets.ImageFolder('data/dogs-vs-cats/test', test_transforms)

BATCH_SIZE = 16

# train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
# valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
# test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

dataloders, dataset_sizes = ImageNetData(data_dir="ImageData",batch_size=BATCH_SIZE,num_workers=0)
train_iterator = dataloders['train']
valid_iterator = dataloders['val']

device = torch.device('cuda')

model = models.resnet50(pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(in_features=2048, out_features=54).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

EPOCHS = 30
SAVE_DIR = 'models/resnet50_pretrain'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet50.pt')

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