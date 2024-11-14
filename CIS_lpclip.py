import torch
import clip
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ClipAdapter import CoCoDataset,Coop,CustomCLIP
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
from sklearn import metrics
from utils.utils import init_random,get_dataloader
device = "cuda:0"
clip_model, preprocess = clip.load("ViT-B/32")
init_random(1)
## ## 
dataloader_shot, dataloader_val, dataloader_test = get_dataloader(
    "CIS",
    train_batch_size=4,
    train_shuffle=True,
    seed=5200,
)
## ##
feature_list = []
label_list = []
for i, (images, target, _) in enumerate(tqdm(dataloader_shot)):
    images= images.to(device)
    img_features= clip_model.encode_image(images)
    feature = img_features.cpu()
    for idx in range(len(images)):
        feature_list.append(feature[idx].tolist())
    label_list.extend(target.tolist())

fewshot_train_feature = feature_list
fewshot_train_label = label_list

feature_list = []
label_list = []
for i, (images, target, _) in enumerate(tqdm(dataloader_val)):
    images= images.to(device)
    img_features= clip_model.encode_image(images)
    feature = img_features.cpu()
    for idx in range(len(images)):
        feature_list.append(feature[idx].tolist())
    label_list.extend(target.tolist())

fewshot_val_feature = feature_list
fewshot_val_label = label_list

search_list = [1e6, 1e4, 1e2, 1, 1e-2, 1e-4, 1e-6]
acc_list = []
for c_weight in search_list:
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_weight).fit(fewshot_train_feature, fewshot_train_label)
    pred = clf.predict(fewshot_val_feature)
    acc_val = sum(pred == fewshot_val_label) / len(fewshot_val_label)
    acc_list.append(acc_val)
print(acc_list)

peak_idx = np.argmax(acc_list)
c_peak = search_list[peak_idx]
c_left, c_right = 1e-1 * c_peak, 1e1 * c_peak


feature_list = []
label_list = []
for i, (images, target, _) in enumerate(tqdm(dataloader_test)):
    images= images.to(device)
    img_features= clip_model.encode_image(images)
    feature = img_features.cpu()
    for idx in range(len(images)):
        feature_list.append(feature[idx].tolist())
    label_list.extend(target.tolist())

test_feature = feature_list
test_label = label_list

clf_left = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_left).fit(fewshot_train_feature, fewshot_train_label)
pred_left = clf_left.predict(fewshot_val_feature)
acc_left = sum(pred_left == fewshot_val_label) / len(fewshot_val_label)
print("Val accuracy (Left): {:.2f}".format(100 * acc_left), flush=True)

clf_right = LogisticRegression(solver="lbfgs", max_iter=1000, penalty="l2", C=c_right).fit(fewshot_train_feature, fewshot_train_label)
pred_right = clf_right.predict(fewshot_val_feature)
acc_right = sum(pred_right == fewshot_val_label) / len(fewshot_val_label)
print("Val accuracy (Right): {:.2f}".format(100 * acc_right), flush=True)

# find maximum and update ranges
if acc_left < acc_right:
    c_final = c_right
    clf_final = clf_right
    # range for the next step
    c_left = 0.5 * (np.log10(c_right) + np.log10(c_left))
    c_right = np.log10(c_right)
else:
    c_final = c_left
    clf_final = clf_left
    # range for the next step
    c_right = 0.5 * (np.log10(c_right) + np.log10(c_left))
    c_left = np.log10(c_left)

pred = clf_final.predict(test_feature)
test_acc = 100 * sum(pred == test_label) / len(pred)
print(test_acc)
