import torch
import clip
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ClipAdapter import CoCoDataset
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import os
from utils.utils import get_dataloader,inin_random

device = "cuda:0"


inin_random(1)

dataloader_shot,dataloader_val,dataloader_test=get_dataloader(type='mocs')

model=resnet50(num_classes=1000,weights=True).to(device)
# model.fc.weight.shape(0)
model.fc=nn.Linear(model.fc.weight.shape[1], len(dataloader_shot.dataset.classnames)).to(device)

def train_model(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    for idx, item in enumerate(tqdm(train_loader)):
        inputs = item[0]
        labels = item[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_corrects += torch.sum(preds.eq(labels))
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
    total_loss = total_loss / total
    acc = 100 * total_corrects / total
    print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, total_loss, acc))
    return total_loss, acc
 
 
def test_model(model, test_loader, loss_fn, optimizer, epoch):
    model.eval()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    with torch.no_grad():
        for idx, item in enumerate(tqdm(test_loader)):
            inputs = item[0]
            labels = item[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds.eq(labels))
 
        loss = total_loss / total
        accuracy = 100 * total_corrects / total
        print("轮次:%4d|训练集损失:%.5f|训练集准确率:%6.2f%%" % (epoch + 1, loss, accuracy))
        return loss, accuracy
 
 
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05)
iter=1000
epoches=int(iter/len(dataloader_shot))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoches//2, gamma=0.1)

for epoch in range(0, epoches):
    loss1, acc1 = train_model(model, dataloader_shot, loss_fn, optimizer, epoch)
    if epoch==epoches-1 or acc1==100:
        loss2, acc2 = test_model(model, dataloader_test, loss_fn, optimizer, epoch)
        break
    scheduler.step() 