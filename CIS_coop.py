import torch
import clip
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ClipAdapter import CoCoDataset, Coop
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn.functional as F
import os
from sklearn import metrics
from utils.utils import get_dataloader, init_random

device = "cuda:0"
clip_model, preprocess = clip.load("ViT-B/32")

init_random(1)
## ##
cls_name = [
    "precast component",
    "precast component delivery truck",
    "bulldozer",
    "dump truck",
    "excavator",
    "concrete mixer",
    "person wearing safety helmet correctly",
    "person who do not wear safety helmet correctly",
    "road roller",
    "wheel loader",
]
dataloader_shot, dataloader_val, dataloader_test = get_dataloader(
    "CIS",
    train_batch_size=4,
    train_shuffle=True,
    seed=5200,
)

## ##
coop = Coop(clip_model, classnames=cls_name).to(device)
for name, param in coop.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)
optimizer = torch.optim.SGD(
    coop.parameters(),
    lr=0.005,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, last_epoch=-1, eta_min=0.005 * 0.01
)
for epoch in tqdm(range(50),desc="Epoch: 50" ):
    all_targets = []
    all_predictions = []
    for i, (images, target, _) in enumerate(dataloader_shot):
        images, target = images.to(device), target.to(device)
        logits = coop(images)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        probs = logits.softmax(dim=-1).cpu()
        pred_label = probs.argmax(dim=1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(pred_label.cpu().numpy())
    scheduler.step()
    acc = metrics.accuracy_score(all_targets, all_predictions)

all_targets = []
all_predictions = []
with torch.no_grad():
    for i, (images, target, _) in enumerate(tqdm(dataloader_test)):
        images = images.to(device)
        logits = coop(images)
        probs = logits.softmax(dim=-1).cpu()
        pred_label = probs.argmax(dim=1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(pred_label.cpu().numpy())
accuracy = metrics.accuracy_score(all_targets, all_predictions)
precision = metrics.precision_score(all_targets, all_predictions, average=None)
recall = metrics.recall_score(all_targets, all_predictions, average=None)
f1 = metrics.f1_score(all_targets, all_predictions, average=None)
print("\n**** CoOp CLIP's val accuracy: {:.2f} ****\n".format(accuracy * 100))
print(precision)
print(recall)
print(f1)
