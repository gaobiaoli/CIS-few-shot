import os
import clip
import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from ClipAdapter import CoCoDataset, ClipAdapter
import argparse
from utils.utils import get_dataloader, init_random

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Save model output to a file with a safe filename."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="ViT-B/32",
        help="The name of the model (e.g., 'vit_l/14')",
    )
    args = parser.parse_args()
    model_name = args.model_name
    init_random(1)
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

    dataloader_shot, dataloader_val, dataloader_test = get_dataloader(
        "mocs",
        train_tranform=train_tranform,
        train_batch_size=4,
        train_shuffle=False,
        seed=34,
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

    clip_adapter.pre_load_features(dataloader=dataloader_test)
    beta, alpha ,_ = clip_adapter.search_hp(
        search_scale=[20, 50], search_step=[200, 20], beta_search=True
    )
    clip_adapter.search_hp(
        search_scale=[20, 51.1], search_step=[200, 51], beta_search=False,print_log=True
    )

    all_predictions, all_targets, (accuracy, precision, recall, f1) = clip_adapter.eval(
        adapt=False
    )
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    print(precision)
    print(recall)
    print(f1)

    all_predictions, all_targets, (accuracy, precision, recall, f1) = clip_adapter.eval(
        adapt=True
    )
    print("\n**** Few-shot CLIP's val accuracy: {:.2f}. ****\n".format(accuracy * 100))
    print(precision)
    print(recall)
    print(f1)
