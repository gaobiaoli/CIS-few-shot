import os
import clip
import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from ClipAdapter import CoCoDataset, ClipAdapter
import argparse 
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save model output to a file with a safe filename.")
    parser.add_argument('--model_name', type=str, required=False, default="ViT-B/32",help="The name of the model (e.g., 'vit_l/14')")
    parser.add_argument('--seed', type=int, required=False, default=34)
    parser.add_argument('--shot', type=int, required=False, default=16)
    args = parser.parse_args()
    model_name = args.model_name
    seed_shot = args.seed
    shot=args.shot

    img_path = "/CV/gaobiaoli/dataset/mocs"
    anno_path = "/CV/gaobiaoli/dataset/mocs"
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # mocstoken = [
    #     "a photo of a worker",
    #     "a photo of a static crane",
    #     "a photo of a hanging head",
    #     "a photo of a crane",
    #     "a photo of a roller",
    #     "a photo of a bulldozer",
    #     "a photo of an excavator",
    #     "a photo of a truck",
    #     "a photo of a loader",
    #     "a photo of a pump truck",
    #     "a photo of a concrete mixer",
    #     "a photo of a pile driving",
    #     "a photo of a household vehicle",
    # ]
    mocstoken = [
        "a photo of a worker",
        "a photo of a tower crane",
        "a photo of a hanging hook",
        "a photo of a vehicle crane",
        "a photo of a roller compactor",
        "a photo of a bulldozer or crawler",
        "a photo of an excavator",
        "a photo of a truck",
        "a photo of a loader",
        "a photo of a concrete pump truck",
        "a photo of a concrete mixer truck",
        "a photo of a pile driver",
        "a photo of a household vehicle",
    ]
    device = "cuda:0"
    clip_model, preprocess = clip.load(model_name, device=device)

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
    coco_json_shot = os.path.join(anno_path, "instances_train.json")
    imgs_path_shot = os.path.join(img_path, "images/train")
    dataset_shot = CoCoDataset(
        coco_json=coco_json_shot,
        imgs_path=imgs_path_shot,
        transform=train_tranform,
        few_shot=shot,
        random_seed=seed_shot,
        category_init_id=1,
    )
    dataloader_shot = DataLoader(
        dataset=dataset_shot, num_workers=4, batch_size=4, shuffle=False
    )
    clip_adapter = ClipAdapter(
        clip_model,
        device=device,
        dataloader=dataloader_shot,
        classnames=mocstoken,
        alpha=5,
        beta=6,
        augment_epoch=10,
    )

    # val
    # coco_json_val = os.path.join(anno_path, "instances_val.json")
    # imgs_path_val = os.path.join(img_path, "images/val")
    # dataset_val = CoCoDataset(
    #     coco_json=coco_json_val,
    #     imgs_path=imgs_path_val,
    #     transform=preprocess,
    #     category_init_id=1,
    # )
    # dataloader_val = DataLoader(
    #     dataset=dataset_val, num_workers=4, batch_size=32, shuffle=False
    # )
    # clip_adapter.pre_load_features(dataloader=dataloader_val)

    # test
    coco_json_test = os.path.join(anno_path, "instances_val.json")
    imgs_path_test = os.path.join(img_path, "images/val")
    dataset_test = CoCoDataset(
        coco_json=coco_json_test,
        imgs_path=imgs_path_test,
        transform=preprocess,
        category_init_id=1,
    )
    dataloader_test = DataLoader(dataset=dataset_test, num_workers=12, batch_size=32)
    clip_adapter.pre_load_features(dataloader=dataloader_test)
    clip_adapter.search_hp(search_scale=[20, 50], search_step=[200, 20],beta_search=True)

    all_predictions, all_targets, (accuracy, precision, recall, f1) = clip_adapter.eval(
        adapt=False
    )
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    # print(precision)
    # print(recall)
    # print(f1)
    
    all_predictions, all_targets, (accuracy, precision, recall, f1) = clip_adapter.eval(
        adapt=True
    )
    print("\n**** Few-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    # print(precision)
    # print(recall)
    # print(f1)
    