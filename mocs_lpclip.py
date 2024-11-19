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
from utils.utils import get_dataloader, init_random
import os
from sklearn import metrics
device = "cuda:0"
clip_model, preprocess = clip.load("ViT-B/32")
init_random(1)
cls_name = [
        "worker",
        "tower crane",
        "hanging hook",
        "vehicle crane",
        "roller compactor",
        "bulldozer",
        "excavator",
        "truck",
        "loader",
        "pump truck",
        "concrete mixer",
        "pile driver",
        "household vehicle",
    ]
## ## 
dataloader_shot, dataloader_val, dataloader_test = get_dataloader(
    "mocs",
    train_tranform=None,
    train_batch_size=4,
    train_shuffle=False,
    seed=34,
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
for i, (images, target, _) in enumerate(tqdm(dataloader_test)):
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
print("Test accuracy : {:.2f}".format(max(acc_list)*100), flush=True)

# all_targets = []
# all_predictions = []
# with torch.no_grad():
#     for i, (images, target, _) in enumerate(tqdm(dataloader_test)):
#         images= images.to(device)
#         logits=coop(images)
#         probs = logits.softmax(dim=-1).cpu()
#         pred_label = probs.argmax(dim=1)
#         all_targets.extend(target.cpu().numpy())
#         all_predictions.extend(pred_label.cpu().numpy())
# accuracy = metrics.accuracy_score(all_targets, all_predictions)
# precision = metrics.precision_score(all_targets, all_predictions, average=None)
# recall = metrics.recall_score(all_targets, all_predictions, average=None)
# f1 = metrics.f1_score(all_targets, all_predictions, average=None)
# print("\n**** CLIP Adapter's val accuracy: {:.2f} ****\n".format(accuracy * 100))
# print(precision)
# print(recall)
# print(f1)