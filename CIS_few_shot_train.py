import os
import torch
import clip
from torch.utils.data import DataLoader
from torchvision import transforms
from ClipAdapter import CoCoDataset, ClipAdapter

if __name__ == "__main__":
    img_path = "/CV/gaobiaoli/dataset/CIS-Dataset"
    anno_path = "/CV/gaobiaoli/dataset/CIS-Dataset/dataset/annotations"
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    CIStoken = [
        "a photo of precast component",
        "a photo of a precast component delivery truck",
        "a photo of a bulldozer",
        "a photo of a dump truck",
        "a photo of an excavator",
        "a photo of a concrete mixer",
        "a photo of a person wearing safety helmet correctly",
        "a photo of a person who do not wear safety helmet correctly",
        "a photo of a road roller",
        "a photo of a wheel loader",
    ]
    device = "cuda:0"
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    # shot
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
    coco_json_shot = os.path.join(anno_path, "train.json")
    imgs_path_shot = os.path.join(img_path, "train")
    dataset_shot = CoCoDataset(
        coco_json=coco_json_shot,
        imgs_path=imgs_path_shot,
        transform=train_tranform,
        few_shot=16,
        random_seed=5200,
        category_init_id=0,
    )
    dataloader_shot = DataLoader(
        dataset=dataset_shot, num_workers=4, batch_size=4, shuffle=False
    )
    clip_adapter = ClipAdapter(
        clip_model,
        device=device,
        dataloader=dataloader_shot,
        classnames=CIStoken,
        alpha=5,
        beta=1,
        augment_epoch=10,
    )

    # val
    coco_json_val = os.path.join(anno_path, "val.json")
    imgs_path_val = os.path.join(img_path, "val")
    dataset_val = CoCoDataset(
        coco_json=coco_json_val,
        imgs_path=imgs_path_val,
        transform=preprocess,
        category_init_id=0,
    )
    dataloader_val = DataLoader(dataset=dataset_val, num_workers=12, batch_size=32)
    clip_adapter.pre_load_features(dataloader=dataloader_val)
    clip_adapter.search_hp(search_scale=[20, 50], search_step=[200, 20],beta_search=False)

    # train
    train_tranform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1),
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
    torch.set_printoptions(precision=2,sci_mode=False)
    dataset_train = CoCoDataset(
        coco_json=coco_json_shot,
        imgs_path=imgs_path_shot,
        transform=train_tranform,
        random_seed=5200,
        ratio=0.1,
        category_init_id=0,
    )
    dataloader_train = DataLoader(
        dataset=dataset_train, num_workers=12, batch_size=256, shuffle=True
    )
    clip_adapter.train_keys(dataloader_train,epoch=20,alpha_train=True,search_hp=False)

    # test
    coco_json_test = os.path.join(anno_path, "test.json")
    imgs_path_test = os.path.join(img_path, "test")
    dataset_test = CoCoDataset(
        coco_json=coco_json_test,
        imgs_path=imgs_path_test,
        transform=preprocess,
        category_init_id=0,
    )
    dataloader_test = DataLoader(dataset=dataset_test, num_workers=12, batch_size=32)
    clip_adapter.pre_load_features(dataloader=dataloader_test)
    (
        all_predictions,
        all_targets,
        (accuracy, precision, recall, f1),
    ) = clip_adapter.eval()
    print("\n**** Few-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    print(precision)
    print(recall)
    print(f1)
    if True:
        clip_adapter.save("./weight/L14.pth")
        import pickle
        with open("./result/L14-train.pkl","wb") as fp:
            pickle.dump([all_predictions,all_targets],fp)