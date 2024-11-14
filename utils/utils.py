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
from clip.clip import _transform


def init_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(type, train_tranform=None, train_shuffle=False,train_batch_size=32,seed=1):
    preprocess = _transform(224)
    if train_tranform is None:
        train_tranform=preprocess
    if type == "mocs":
        img_path = "/CV/gaobiaoli/dataset/mocs"
        anno_path = "/CV/gaobiaoli/dataset/mocs"
        coco_json_shot = os.path.join(anno_path, "instances_train.json")
        imgs_path_shot = os.path.join(img_path, "images/train")
        coco_json_val = os.path.join(anno_path, "instances_val.json")
        imgs_path_val = os.path.join(img_path, "images/val")
        coco_json_test = os.path.join(anno_path, "instances_val.json")
        imgs_path_test = os.path.join(img_path, "images/val")
        category_init_id = 1

    elif type == "CIS":
        img_path = "/CV/gaobiaoli/dataset/CIS-Dataset"
        anno_path = "/CV/gaobiaoli/dataset/CIS-Dataset/dataset/annotations"
        coco_json_shot = os.path.join(anno_path, "train.json")
        imgs_path_shot = os.path.join(img_path, "train")
        coco_json_val = os.path.join(anno_path, "val.json")
        imgs_path_val = os.path.join(img_path, "val")
        coco_json_test = os.path.join(anno_path, "test.json")
        imgs_path_test = os.path.join(img_path, "test")
        category_init_id = 0

    dataset_shot = CoCoDataset(
        coco_json=coco_json_shot,
        imgs_path=imgs_path_shot,
        transform=train_tranform,
        few_shot=16,
        random_seed=seed,
        category_init_id=category_init_id,
    )
    dataloader_shot = DataLoader(
        dataset=dataset_shot, num_workers=4, batch_size=train_batch_size, shuffle=train_shuffle
    )

    dataset_val = CoCoDataset(
        coco_json=coco_json_val,
        imgs_path=imgs_path_val,
        transform=preprocess,
        category_init_id=category_init_id,
    )
    dataloader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=32)
    
    dataset_test = CoCoDataset(
        coco_json=coco_json_test,
        imgs_path=imgs_path_test,
        transform=preprocess,
        category_init_id=category_init_id,
    )
    dataloader_test = DataLoader(dataset=dataset_test, num_workers=12, batch_size=32)
    return dataloader_shot,dataloader_val,dataloader_test