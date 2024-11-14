import torch
import clip
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ClipAdapter import CoCoDataset,DirDataset
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import os
device = "cuda:0"
clip_model, preprocess = clip.load("ViT-B/16", device=device)
del clip_model
root=r'/CV/gaobiaoli/dataset/rebar_tying'
class_dir_map={
    "a photo of a worker squatting or bending to tie steel bars":"12",
    "a photo of a worker doing non-rebar work or taking a break":"3",
}
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
train_tranform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.9, 1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
shot_num=20
dataset_shot=DirDataset(root=root,class_dir_map=class_dir_map,transform=train_tranform,few_shot=16,random_seed=seed)
dataloader_shot=DataLoader(dataset=dataset_shot,batch_size=shot_num,num_workers=4,shuffle=False)
dataset_test=DirDataset(root=root,class_dir_map=class_dir_map,transform=train_tranform,few_shot=16,random_seed=seed,reverse=True)
dataloader_test=DataLoader(dataset=dataset_test,batch_size=shot_num,num_workers=4,shuffle=True)
model=resnet50(num_classes=1000,weights=True).to(device)
# model.fc.weight.shape(0)
model.fc=nn.Linear(model.fc.weight.shape[1], 2).to(device)
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
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
epoches=10
for epoch in range(0, epoches):
    loss1, acc1 = train_model(model, dataloader_shot, loss_fn, optimizer, epoch)
    if epoch==epoches-1:
        loss2, acc2 = test_model(model, dataloader_test, loss_fn, optimizer, epoch)
    scheduler.step() 
model.train()